import os
import pika
import pickle
import argparse
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import requests
import random
import copy
import gzip

from tqdm import tqdm
from collections import deque, OrderedDict
from requests.auth import HTTPBasicAuth

import src.Validation
import src.Log
import src.Model

from src.Utils import calculate_md, train_gmm_model, verify_gradient
from src.Utils import krum, get_weight_vector
from src.Utils import state_dict_to_vector, byzantine_tolerance_aggregation, median_aggregation, trimmed_mean_aggregation 
from src.Utils import cosine, DBSCAN_phase2
from src.Utils import flatten_state_dict, fltracer_detect_anomalies
from src.Utils import quantize_vector, l2_norm, dequantize, cosine_distance
from src.Model import ICUData

from client import*
from torch.utils.data import Subset



parser = argparse.ArgumentParser(description="Federated learning framework with controller.")

parser.add_argument('--device', type=str, required=False, help='Device of server')

args = parser.parse_args()

device = None

if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

total_clients = config["server"]["clients"]
model_name = config["server"]["model"]
data_name = config["server"]["data-name"]
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
num_round = config["server"]["num-round"]
load_parameters = config["server"]["parameters"]["load"]
validation = config["server"]["validation"]
genuine_rate = config["server"]["genuine-rate"]
random_seed = config["server"]["random-seed"]

data_distribution = config["server"]["data-distribution"]
server_mode = config["server"]["mode"]
data_range = data_distribution["num-data-range"]

hyper_detection = config["server"]["hyper-detection"]
hyper_detection_enable = hyper_detection["enable"]
cosine_search = hyper_detection["cosine-search"]
DBSCAN_n_components = hyper_detection["n_components"]
DBSCAN_eps = hyper_detection["eps"]
DBSCAN_min_samples = hyper_detection["min_samples"]

# Clients
epoch = config["learning"]["epoch"]
batch_size = config["learning"]["batch-size"]
lr = config["learning"]["learning-rate"]
hyper_lr = config["learning"]["hyper-lr"]
momentum = config["learning"]["momentum"]
clip_grad_norm = config["learning"]["clip-grad-norm"]

log_path = config["log_path"]

if data_name == "CIFAR10" or data_name == "MNIST":
    num_labels = 10
else:
    num_labels = 0

if random_seed:
    random.seed(random_seed)


class Server:
    def __init__(self):
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        self.channel = self.connection.channel()
        self.num_round = num_round
        self.round = self.num_round

        self.channel.queue_declare(queue='rpc_queue')
        self.genuine_rate = 0.0
        self.total_clients = total_clients
        self.current_clients = 0
        self.updated_clients = 0
        self.responses = {}  # Save response
        self.list_clients = []
        self.list_attack_clients = []
        self.all_model_parameters = []
        self.all_genuine_parameters = []
        self.final_state_dict = None
        self.round_result = True
        self.byzantine_ratio = 0.0
        self.selected_client = []

        # Initial hyper training model
        self.net = None
        self.hnet = None
        
        # Training FLTrust
        self.model = None
        self.root_loader = None

        # detection algorithms
        self.previous_hyper = None
        self.cosine_search = 5
        self.list_embeddings = [deque(maxlen=self.cosine_search) for _ in range(self.total_clients)]
  

        if server_mode == "hyper":
            if not self.net and not self.hnet:
                if hasattr(src.Model, model_name):
                    self.net = getattr(src.Model, model_name)().to(device)
                else:
                    raise ValueError(f"Model name '{model_name}' is not valid.")

                if load_parameters:
                    filepath_hyper = f'{model_name}_hyper_{total_clients}.pth'
                    filepath = f'{model_name}.pth'
                    if os.path.exists(filepath_hyper):
                        # if contain hyper model
                        src.Log.print_with_color(f"Load state dict from hyper model: {filepath_hyper}", "yellow")
                        state_dict = torch.load(filepath_hyper, weights_only=True)
                        self.load_new_hyper()
                        self.hnet.load_state_dict(state_dict)
                    elif os.path.exists(filepath):
                        # if contain initial model
                        src.Log.print_with_color(f"Load state dict from original model: {filepath}", "yellow")
                        state_dict = torch.load(filepath, weights_only=True)
                        self.net.load_state_dict(state_dict)
                        self.load_new_hyper()
                    else:
                        self.load_new_hyper()
                else:
                    self.load_new_hyper()
                self.load_new_hyper()

            self.optimizer = torch.optim.Adam(self.hnet.parameters(), lr=hyper_lr)

        if server_mode == "FLTrust":
            if not self.model:
                if hasattr(src.Model, model_name):
                    self.model = getattr(src.Model, model_name)().to(device)
                else:
                    raise ValueError(f"Model name '{model_name}' is not valid.")
        
        
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.validation = src.Validation.Validation(model_name, data_name, self.logger)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        self.logger.log_info("### Application start ###\n")
        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        """
        Response message to clients
        :param client_id: client ID
        :param message: message
        :return:
        """
        reply_channel = self.channel
        reply_queue_name = f'reply_{client_id}'
        reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def on_request(self, ch, method, props, body):
        """
        Handler request from clients
        :param ch: channel
        :param method:
        :param props:
        :param body: message body
        :return:
        """
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            attack = message["attack"]

            if str(client_id) not in self.list_clients:
                if attack:
                    self.list_attack_clients.append(str(client_id))

                self.list_clients.append(str(client_id))
                src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")

            # If consumed all clients - Register for first time
            if len(self.list_clients) == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.client_selection()
                src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                self.notify_clients()
        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            src.Log.print_with_color(f"[<<<] Received message from client: {data_message}", "blue")
            self.updated_clients += 1
            # Save client's model parameters
            if not result:
                self.round_result = False

            if self.round_result:
                if server_mode == "FLTrust":
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    if self.global_model_state_dict is None:
                        self.global_model_state_dict = copy.deepcopy(self.model.state_dict())
                    # Tính update: wR - w0
                    delta = {k: model_state_dict[k].to(device) - self.global_model_state_dict[k].to(device) for k in model_state_dict}

                    self.all_model_parameters.append({
                        'client_id': client_id,
                        'weight': delta,  # lưu gradient update
                        'size': client_size
                    })
                    if str(client_id) not in self.list_attack_clients:
                        self.all_genuine_parameters.append(model_state_dict)
             
                else:        
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                    'size': client_size})
                    if str(client_id) not in self.list_attack_clients:
                        self.all_genuine_parameters.append(model_state_dict)

            # If consumed all client's parameters
            if self.updated_clients == len(self.selected_client):
                self.process_consumer()

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def process_consumer(self):
        """
        After collect all training clients, start validation and make decision for the next training round
        :return:
        """
        self.updated_clients = 0
        src.Log.print_with_color("Collected all parameters.", "yellow")    

        # TODO: detect model poisoning with self.all_model_parameters at here
        if self.round_result:
            if server_mode == "fedavg":
                self.avg_all_parameters()
            if server_mode == "FLTrust":
                with gzip.open("data/test_dataset.pkl.gz", "rb") as f:
                    root_dataset = pickle.load(f)
                D0 = Subset(root_dataset, range(200))
                self.root_loader = torch.utils.data.DataLoader(D0 , batch_size=100, shuffle=False)
                self.train_FLTrust()
            elif server_mode == "hyper":
                src.Log.print_with_color(f"Start training hyper model!", "yellow")
                if hyper_detection_enable:
                    self.previous_hyper = copy.deepcopy(self.hnet.state_dict())
                self.train_hyper()
                
            elif server_mode == "trimmed_mean":
                src.Log.print_with_color(f"Using  trimmed mean aggregation!", "yellow")
                self.final_state_dict = trimmed_mean_aggregation(
                    [model['weight'] for model in self.all_model_parameters]
                )
            elif server_mode == "shieldfl":
                src.Log.print_with_color(f"Using ShieldFL aggregation!", "yellow")
                print(f"Number of model weight = {len(self.all_model_parameters)}")
            
                client_vectors = []
                for entry in self.all_model_parameters:
                    vec = state_dict_to_vector(entry['weight'])
                    vec = vec / (vec.norm() + 1e-8)  # normalize
                    client_vectors.append(vec)
            
                # Tham chiếu trung bình
                ref = sum(client_vectors) / len(client_vectors)
            
                # Tính độ tương đồng cosine với vector ref
                cosine_sims = torch.tensor([
                    F.cosine_similarity(vec.view(1, -1), ref.view(1, -1)).item()
                    for vec in client_vectors
                ])
            
                # Độ lệch là (1 - cosine_similarity)
                deviation = 1 - cosine_sims
            
                # Trọng số ngược lệch → client càng gần ref thì weight càng lớn
                weights = 1 / (deviation + 1e-6)  # thêm epsilon tránh chia 0
                weights = weights / weights.sum()  # chuẩn hóa tổng = 1
            
                # Weighted aggregation
                state_dict_list = [entry['weight'] for entry in self.all_model_parameters]
                avg_state = copy.deepcopy(state_dict_list[0])  # deepcopy để giữ nguyên
            
                for key in avg_state.keys():
                    if torch.is_floating_point(avg_state[key]):
                        avg_state[key] = sum(
                            state_dict_list[i][key].float() * weights[i]
                            for i in range(len(state_dict_list))
                        )
                    elif avg_state[key].dtype in [torch.long, torch.int, torch.int64]:
                        avg_state[key] = sum(
                            state_dict_list[i][key] * weights[i]
                            for i in range(len(state_dict_list))
                        ).round().to(torch.long)  # làm tròn về long
                    else:
                        continue  # bỏ qua kiểu khác

                self.final_state_dict = avg_state
                
            elif server_mode == "gmm":
                src.Log.print_with_color(f"Using GMM-based gradient filtering!", "yellow")
                
                benign_gradients = [get_weight_vector(client['weight']) for client in self.all_model_parameters if client['client_id'] not in self.list_attack_clients]
                malicious_gradients = [get_weight_vector(client['weight']) for client in self.all_model_parameters if client['client_id'] in self.list_attack_clients]

                gmm_model = train_gmm_model(benign_gradients, malicious_gradients, n_components=2)

                filtered_weights = []
                threshold = 3 * np.std([calculate_md(g, gmm_model.means_[0], gmm_model.covariances_[0]) for g in benign_gradients])
                
                for client in self.all_model_parameters:
                    gradient = get_weight_vector(client['weight'])
                    result = verify_gradient(gradient, gmm_model, threshold)
                    if result == "benign":
                        filtered_weights.append(client['weight'])

                if filtered_weights:
                    self.final_state_dict = self.avg_selected_parameters(filtered_weights)
                else:
                    self.round_result = False
            elif server_mode == "krum":
                src.Log.print_with_color("Using Krum aggregation!", "yellow")
                # Chuyển state_dict thành vector flatten
                client_vectors = []
                state_dict_map = {}

                for client in self.all_model_parameters:
                    vec = get_weight_vector(client['weight'])  # flatten model
                    client_vectors.append(torch.tensor(vec, dtype=torch.float32))
                    state_dict_map[vec.tobytes()] = client['weight']  # để ánh xạ lại

                f = int(len(client_vectors) * self.genuine_rate)

                selected_vector = krum(client_vectors, f)
                selected_state_dict = state_dict_map[selected_vector.numpy().tobytes()]

                self.final_state_dict = selected_state_dict
            elif server_mode == "median":
                src.Log.print_with_color(f"Using Median aggregation!", "yellow")
                self.final_state_dict = median_aggregation(
                    [model['weight'] for model in self.all_model_parameters]
                )
            # elif server_mode == "fltracer":
            #     src.Log.print_with_color(f"Using FLTracer aggregation!", "yellow")
            #     print(f"Number of model weight = {len(self.all_model_parameters)}")
            
            #     state_dict_list = [entry['weight'] for entry in self.all_model_parameters]
            #     client_sizes = [entry['size'] for entry in self.all_model_parameters]
            
            #     flat_weights = [flatten_state_dict(sd) for sd in state_dict_list]
            #     weight_matrix = np.stack([w.numpy() for w in flat_weights])
            
            #     anomaly_indices = fltracer_detect_anomalies(weight_matrix)
            #     print(f"Anomaly indices = {anomaly_indices}")
            
            #     # Lọc ra những client không bị nghi ngờ
            #     benign_entries = [
            #         entry for i, entry in enumerate(self.all_model_parameters)
            #         if i not in anomaly_indices
            #     ]
            #     # benign_entries = self.all_model_parameters
            
            #     benign_state_dicts = [entry['weight'] for entry in benign_entries]
            #     benign_sizes = [entry['size'] for entry in benign_entries]
            #     total_size = sum(benign_sizes)
                
            #     global_model_state_dict = copy.deepcopy(benign_state_dicts[0])  # cần deepcopy để không làm thay đổi bản gốc
            
            #     for key in global_model_state_dict.keys():
            #         if torch.is_floating_point(global_model_state_dict[key]):
            #             global_model_state_dict[key] = sum(
            #                 benign_state_dicts[i][key] * benign_sizes[i]
            #                 for i in range(len(benign_state_dicts))
            #             ) / total_size
            #         elif global_model_state_dict[key].dtype in [torch.long, torch.int, torch.int64]:
            #             global_model_state_dict[key] = sum(
            #                 benign_state_dicts[i][key] * benign_sizes[i]
            #                 for i in range(len(benign_state_dicts))
            #             ) // total_size
            #         else:
            #             continue  # Bỏ qua kiểu dữ liệu không hỗ trợ
            
            #     self.final_state_dict = global_model_state_dict
            elif server_mode == "scionfl":
                src.Log.print_with_color(f"Using ScionFL aggregation!", "yellow")
                print(f"Number of model weight = {len(self.all_model_parameters)}")
            
                state_dict_list = [entry['weight'] for entry in self.all_model_parameters]
                client_sizes = [entry['size'] for entry in self.all_model_parameters]
            
                flat_vectors = [flatten_state_dict(sd) for sd in state_dict_list]
                quantized_updates = [quantize_vector(vec) for vec in flat_vectors]
            
                # Step 1: L2 norm clipping
                l2_values = [l2_norm(*q) for q in quantized_updates]
                l2_avg = np.mean(l2_values)
                MU_THRESHOLD = 3
                TOPK_RATIO = 0.5
            
                clipped_updates = []
                for (sigma, smin, smax), l2 in zip(quantized_updates, l2_values):
                    if l2 > MU_THRESHOLD * l2_avg:
                        factor = (MU_THRESHOLD * l2_avg) / l2
                        smin *= factor
                        smax *= factor
                    clipped_updates.append((sigma, smin, smax))
            
                # Step 2: Dequantize + tính vector trung bình
                agg_vector = sum(dequantize(*q) for q in clipped_updates) / len(clipped_updates)
            
                # Step 3: Cosine filtering
                cosine_scores = [cosine_distance(dequantize(*q), agg_vector) for q in clipped_updates]
                threshold = sorted(cosine_scores, reverse=True)[int(TOPK_RATIO * len(cosine_scores))]
                benign_indices = [i for i, s in enumerate(cosine_scores) if s > threshold]
            
                print(f"Benign indices = {benign_indices}")
            
                # Step 4: Lấy state_dict và size của các client hợp lệ
                benign_state_dicts = [state_dict_list[i] for i in benign_indices]
                benign_sizes = [client_sizes[i] for i in benign_indices]
                total_size = sum(benign_sizes)
            
                # Step 5: Tổng hợp theo kiểu FedAvg (có trọng số)
                new_sd = copy.deepcopy(benign_state_dicts[0])  # sao chép state_dict đầu tiên
            
                for key in new_sd.keys():
                    if torch.is_floating_point(new_sd[key]):
                        new_sd[key] = sum(
                            benign_state_dicts[i][key].float() * benign_sizes[i]
                            for i in range(len(benign_state_dicts))
                        ) / total_size
                    elif new_sd[key].dtype in [torch.long, torch.int, torch.int64]:
                        new_sd[key] = sum(
                            benign_state_dicts[i][key] * benign_sizes[i]
                            for i in range(len(benign_state_dicts))
                        ).round().to(torch.long)
                    else:
                        continue  # bỏ qua kiểu dữ liệu khác
            
                self.final_state_dict = new_sd
            else:
                raise ValueError(f"Server mode '{server_mode}' is not valid.")

        if hyper_detection_enable:
        
            to_remove = []
            for i in self.selected_client:
                # _, current_embedding = self.hnet(torch.tensor([i], dtype=torch.long).to(device))
                # current_embedding = current_embedding.reshape(-1).tolist()
                # current_embedding = np.array(current_embedding)
                
                _, current_embedding = self.hnet(torch.tensor([i], dtype=torch.long).to(device))
                current_embedding = np.array(current_embedding.detach().cpu())
                current_embedding = current_embedding.reshape(1, -1)
                # convert client indices to client id
                # client_id = self.list_clients[i]

                #previous_embeddings = list(self.list_embeddings[i])
                previous_embeddings = np.vstack(self.list_embeddings[i]) if self.list_embeddings[i] else np.empty((0, current_embedding.shape[1]))
                
                if self.num_round - self.round + 1 >= 18:
                    if cosine(previous_embeddings, current_embedding):
                        to_remove.append(i)

                self.list_embeddings[i].append(current_embedding)
               
                embeddings_to_save = [list(dq) for dq in self.list_embeddings]

                # Lưu với object dtype để hỗ trợ list không đồng đều
                np.save("all_embeddings.npy", np.array(embeddings_to_save, dtype=object))
                
            if self.num_round - self.round + 1 >= 18:
                  
            # embedding round n and n-1
                outliers = DBSCAN_phase2([sublist[-2] for sublist in self.list_embeddings],
                                        [sublist[-1] for sublist in self.list_embeddings],
                                        self.selected_client, n_components=DBSCAN_n_components, eps=DBSCAN_eps, min_samples=DBSCAN_min_samples)
                print("Outliers từ DBSCAN:", outliers)
                to_remove = list(set(to_remove) & set(outliers))
                if to_remove:
                    for node_id in to_remove:
                        print(f"Removing anomaly {node_id}, rolling back")
                        self.selected_client.remove(node_id)
                    self.hnet.load_state_dict(self.previous_hyper)

        # Server validation
        if validation and self.round_result:
            if server_mode == "hyper":
                self.round_result = self.validation.test_hyper(self.hnet, len(self.all_model_parameters), device)
            else:
                self.round_result = self.validation.test(self.final_state_dict, device)

        self.all_model_parameters = []
        if not self.round_result:
            src.Log.print_with_color(f"Training failed!", "yellow")
        else:
            # Save to files
            if server_mode == "hyper":
                torch.save(self.hnet.state_dict(), f'{model_name}_hyper_{total_clients}.pth')
            else:
                torch.save(self.final_state_dict, f'{model_name}.pth')

            self.round -= 1

        self.round_result = True

        if self.round > 0:
            # Start a new training round
            src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
            # self.client_selection()
            self.notify_clients()
        else:
            # Stop training
            self.notify_clients(start=False)
            sys.exit()

    def notify_clients(self, start=True):
        """
        Control message to clients
        :param start: If True (default), request clients to start. Else if False, stop training
        :return:
        """
        # Send message to clients when consumed all clients
        self.global_model_state_dict = self.final_state_dict
        if start:
            # Read parameters file
            state_dict = None
            if server_mode != "hyper":
                if load_parameters:
                    filepath = f'{model_name}.pth'
                    if os.path.exists(filepath):
                        state_dict = torch.load(filepath)
                else:
                    state_dict = self.final_state_dict

            for i in self.selected_client:
                if server_mode == "hyper":
                    state_dict, _ = self.hnet(torch.tensor([i], dtype=torch.long).to(device))
                # convert client indices to client id
                client_id = self.list_clients[i]
                # Request clients to start training
                src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")

                genuine_models = None
                if client_id in self.list_attack_clients:
                    if len(self.all_genuine_parameters) > 0:
                        genuine_models = random.sample(self.all_genuine_parameters,
                                                       max(int(genuine_rate * len(self.all_genuine_parameters)), 1))

                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "model_name": model_name,
                            "data_name": data_name,
                            "parameters": state_dict,
                            "data_ranges": data_range,
                            "epoch": epoch,
                            "batch_size": batch_size,
                            "lr": lr,
                            "momentum": momentum,
                            "clip_grad_norm": clip_grad_norm,
                            "genuine_models": genuine_models}
                self.send_to_response(client_id, pickle.dumps(response))
            # clear all genuine models
            self.all_genuine_parameters = []
        else:
            for client_id in self.list_clients:
                # Request clients to stop process
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!",
                            "parameters": None}
                self.send_to_response(client_id, pickle.dumps(response))

    def client_selection(self):
        """
        Select the specific clients
        :return: The list contain index of active clients: `self.selected_client`.
        E.g. `self.selected_client = [2,3,5]` means client 2, 3 and 5 will train this current round
        """
        self.selected_client = [i for i in range(len(self.list_clients))]

        # From client selected, calculate and log training time
        self.logger.log_info(f"Active with {len(self.selected_client)} client: {self.selected_client}")

    def train_hyper(self):
        """
        Consuming all client's weight from `self.all_model_parameters` and start training hyper model
        :return: Global weight on `self.final_state_dict`
        """
        self.hnet.train()
        
        for node_id in tqdm(self.selected_client):
            weights, _ = self.hnet(torch.tensor([node_id]).to(device))

            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})
            self.optimizer.zero_grad()
            final_state = self.find_weight(node_id)         # get client's weight
            final_state = {k: v.to(device) for k, v in final_state.items()}
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

            # Calculate gradients with respect to hypernetwork parameters
            hnet_grads = torch.autograd.grad(
                outputs=list(weights.values()),
                inputs=self.hnet.parameters(),
                grad_outputs=list(delta_theta.values()),
                allow_unused=True
            )

            # Update the hypernetwork parameters
            for p, g in zip(self.hnet.parameters(), hnet_grads):
                # Check if gradient is not None
                # if g is not None:
                p.grad = g

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), clip_grad_norm)
            
            self.optimizer.step()

        # Update new clients' weight
        src.Log.print_with_color(f"Update new clients' weight", "yellow")
        for i in range(len(self.all_model_parameters)):
            c = self.list_clients.index(str(self.all_model_parameters[i]["client_id"]))
            print(f"Get client on index {c}: client id {self.all_model_parameters[i]['client_id']}")
            model_dict, _ = self.hnet(torch.tensor([c]).to(device))
            self.all_model_parameters[i]["weight"] = model_dict

        # self.avg_all_parameters()
    #Train FLTrust 
    @staticmethod
    def cosine_similarity(u, v):
    # Nếu là dict (state_dict), convert sang list tensor
        if isinstance(u, dict):
            u = [p.data.clone() for p in u.values()]
        if isinstance(v, dict):
            v = [p.data.clone() for p in v.values()]

        u_flat = torch.cat([p.flatten() for p in u])
        v_flat = torch.cat([p.flatten() for p in v])

        return torch.nn.functional.cosine_similarity(u_flat, v_flat, dim=0).item()

    @staticmethod
    def normalize_update(update_dict, scale):
        return {k: v * scale for k, v in update_dict.items()}

    @staticmethod
    def subtract_state_dict(wR, w0):
        return {k: wR[k] - w0[k] for k in wR}

    def train_FLTrust(self):
        # g_0 = self.model.state_dict()
        if self.final_state_dict:
            g_0 = self.final_state_dict
            self.model.load_state_dict(self.final_state_dict)
        else:
            g_0 = self.model.state_dict()
            
        train_on_device(self.model, data_name, epoch, lr, momentum, clip_grad_norm, self.root_loader)
        g0 = self.model.state_dict()
        g0_delta = self.subtract_state_dict(g0, g_0)
        norm_g0 = torch.sqrt(sum(torch.norm(p) ** 2 for p in g0_delta.values()))
        
        weighted_updates = None
        total_weight = 0.0
        self.trust_scores = {}
        self.updates = {}

        for node_id in tqdm(self.selected_client):
            local_model = self.find_weight(node_id)
            local_delta = self.subtract_state_dict(local_model, g_0)
            norm_gi = torch.sqrt(sum(torch.norm(p) ** 2 for p in local_delta.values()))
            trust_score = max(0.0, self.cosine_similarity(local_delta, g0_delta))  # ReLU(cos_sim)
            scale = (norm_g0 / (norm_gi + 1e-6)) * trust_score
            scaled_update = self.normalize_update(local_delta, scale)
            # Cộng dồn update
            if weighted_updates is None:
                weighted_updates = {k: v.clone() for k, v in scaled_update.items()}
            else:
                for k in weighted_updates:
                    weighted_updates[k] += scaled_update[k]
            total_weight += trust_score
            self.trust_scores[node_id] = trust_score
            self.updates[node_id] = scaled_update

        # Final global update
        for k in weighted_updates:
            weighted_updates[k] /= (total_weight + 1e-6)
        # Áp dụng global update vào model
        new_state_dict = {k: g_0[k] + weighted_updates[k] for k in g_0}
        self.final_state_dict = new_state_dict  
        
    def find_weight(self, node_id):
        for w in self.all_model_parameters:
            if str(w["client_id"]) == self.list_clients[node_id]:
                return w["weight"]
        src.Log.print_with_color(f"[Warning] Cannot find weight of node id {node_id}!", "yellow")

    def avg_all_parameters(self):
        """
        Consuming all client's weight from `self.all_model_parameters` - a list contain all client's weight
        :return: Global weight on `self.final_state_dict`
        """
        # Average all client parameters
        num_models = len(self.all_model_parameters)
        src.Log.print_with_color(f"Number of models' parameters = {num_models}", "yellow")

        if num_models == 0:
            return

        self.final_state_dict = self.all_model_parameters[0]['weight']
        all_client_sizes = [item['size'] for item in self.all_model_parameters]

        for key in self.final_state_dict.keys():
            if self.final_state_dict[key].dtype != torch.long:
                self.final_state_dict[key] = sum(self.all_model_parameters[i]['weight'][key] * all_client_sizes[i]
                                                 for i in range(num_models)) / sum(all_client_sizes)
            else:
                self.final_state_dict[key] = sum(self.all_model_parameters[i]['weight'][key] * all_client_sizes[i]
                                                 for i in range(num_models)) // sum(all_client_sizes)

        if not self.final_state_dict:
            src.Log.print_with_color(f"[Warning] Final state dict is None!", "yellow")

    def avg_selected_parameters(self, selected_weights):
        """
        Tính trung bình các mô hình được chọn (sau khi lọc bằng GMM).
        :param selected_weights: List các state_dict đã qua lọc.
        :return: state_dict trung bình.
        """
        num_models = len(selected_weights)
        if num_models == 0:
            src.Log.print_with_color("Không có mô hình hợp lệ sau khi lọc GMM!", "red")
            return None

        # Copy mô hình đầu tiên làm mẫu
        avg_weights = {k: selected_weights[0][k].clone() for k in selected_weights[0]}

        for key in avg_weights.keys():
            if avg_weights[key].dtype != torch.long:
                avg_weights[key] = sum(model[key] for model in selected_weights) / num_models
            else:
                avg_weights[key] = sum(model[key] for model in selected_weights) // num_models

        return avg_weights

    def load_new_hyper(self):
        self.hnet = src.Model.HyperNetwork(self.net, self.total_clients, 8, 100, False, 2).to(device)
        #self.hnet = src.Model.CNNHyper(self.total_clients, 10 , 100, 3).to(device)
        
def delete_old_queues():
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "gradient_queue"):
                try:
                    http_channel.queue_delete(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to delete queue '{queue_name}': {e}", "yellow")
            else:
                try:
                    http_channel.queue_purge(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' purged.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to purge queue '{queue_name}': {e}", "yellow")

        connection.close()
        return True
    else:
        src.Log.print_with_color(
            f"Failed to fetch queues from RabbitMQ Management API. Status code: {response.status_code}", "yellow")
        return False


if __name__ == "__main__":
    delete_old_queues()
    server = Server()
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
