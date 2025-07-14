import time
import pickle
import pika
import torch
import random
import gzip
import pandas as pd

from pika.exceptions import AMQPConnectionError

import src.Log
import src.Model
import src.Utils


class RpcClient:
    def __init__(self, client_id, address, username, password, train_func, device):
        self.model = None
        self.client_id = client_id
        self.address = address
        self.username = username
        self.password = password
        self.train_func = train_func
        self.device = device
        self.training_round = 0

        self.attack = False
        self.attack_mode = None
        self.attack_round = 0
        self.attack_args = None
        self.genuine_models = None

        self.channel = None
        self.connection = None
        self.response = None

        self.train_set = None
        self.label_to_indices = None

        self.connect()

    def wait_response(self):
        status = True
        reply_queue_name = f'reply_{self.client_id}'
        self.channel.queue_declare(reply_queue_name, durable=False)
        while status:
            try:
                method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
                if body:
                    status = self.response_message(body)
                time.sleep(0.5)
                
            except AMQPConnectionError as e:
                print(f"Connection failed, retrying in 5 seconds: {e}")
                self.connect()
                time.sleep(5)

    def set_attack_config(self, mode, start_round, attack_args):
        self.attack = True
        self.attack_mode = mode
        self.attack_round = start_round
        self.attack_args = attack_args

    def response_message(self, body):
        self.response = pickle.loads(body)
        src.Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]
        state_dict = self.response["parameters"]

        if action == "START":
            model_name = self.response["model_name"]
            self.training_round += 1
            if self.model is None:
                if hasattr(src.Model, model_name):
                    self.model = getattr(src.Model, model_name)()
                else:
                    raise ValueError(f"Model name '{model_name}' is not valid.")

            # Read parameters and load to model
            if state_dict:
                try:
                    # for name, param in self.model.named_parameters():
                    #     print(name, param.data.shape)
                    # self.model.load_state_dict(state_dict, strict=False)
                    self.model.load_state_dict(state_dict)
                except Exception as e:
                    src.Log.print_with_color(f"[Warning] Cannot load state dict on client. {e}", "yellow")
                    # Instead of loading weights directly, update net's parameters using the generated weights
                    for name, param in self.model.named_parameters():
                        param.data = state_dict[name]

            data_ranges = self.response["data_ranges"]
            genuine_models = self.response["genuine_models"]
            if self.attack and genuine_models:
                self.genuine_models = genuine_models

            num_data = random.randrange(data_ranges[0], data_ranges[1] + 1)
            src.Log.print_with_color(f"Number of client's data: {num_data}", "yellow")

            if self.attack and self.training_round >= self.attack_round and self.genuine_models and len(self.genuine_models) > 0:
                src.Log.print_with_color(f"[===] Client is start attacking {self.attack_mode} ...", "red")
                # Apply difference attack algorithm here
                src.Log.print_with_color(f"[<<<] Attacker received {len(self.genuine_models)} genuine models", "blue")
                result, model_state_dict = self.malicious_training(self.genuine_models)
            else:
                result, model_state_dict = self.genuine_training(num_data)

            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')
            data = {"action": "UPDATE", "client_id": self.client_id, "result": result, "size": num_data,
                    "message": "Sent parameters to Server", "parameters": model_state_dict}
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True
        elif action == "STOP":
            return False

    def malicious_training(self, genuine_models):
        if self.attack_mode == "Random":
            base_model = src.Utils.create_random_base_model(self.model.state_dict(), perturbation=self.attack_args[0])
            return True, base_model
        elif self.attack_mode == "Min-Max":
            attack_model = src.Utils.create_min_max_model(self.model.to('cpu').state_dict(), genuine_models)
            if attack_model:
                return True, attack_model
            else:
                return False, None
        elif self.attack_mode == "Min-Sum":
            attack_model = src.Utils.create_min_sum_model(self.model.to('cpu').state_dict(), genuine_models)
            if attack_model:
                return True, attack_model
            else:
                return False, None
        elif self.attack_mode == "Opt-Fang":
            attack_model = src.Utils.create_opt_fang_model(self.model.to('cpu').state_dict(), genuine_models)
            if attack_model:
                return True, attack_model
            else:
                return False, None
        elif self.attack_mode == "LIE":
            attack_model = src.Utils.create_LIE_state_dict(genuine_models, scaling_factor=self.attack_args[0])
            return True, attack_model
        else:
            raise ValueError(f"Attack client not contain '{self.attack_mode}' algorithm.")

    def genuine_training(self, num_data):
        data_name = self.response["data_name"]
        batch_size = self.response["batch_size"]
        lr = self.response["lr"]
        momentum = self.response["momentum"]
        epoch = self.response["epoch"]
        clip_grad_norm = self.response["clip_grad_norm"]

        if data_name and not self.train_set:
            
            if data_name == "HAR":
                with gzip.open("data/icu_har_train_ds.pkl.gz", "rb") as f:
                    self.train_set = pickle.load(f)
            if data_name == "ICU":
                with gzip.open("train_dataset.pkl.gz", "rb") as f:
                    self.train_set = pickle.load(f)
            else:
                raise ValueError(f"Data name '{data_name}' is not valid.")

        selected_indices = random.sample(range(len(self.train_set)), num_data)
        subset = torch.utils.data.Subset(self.train_set, selected_indices)

        train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        
        # Stop training, then send parameters to server
        return self.train_func(self.model, data_name, epoch, lr, momentum, clip_grad_norm, train_loader), self.model.state_dict()

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
        self.channel = self.connection.channel()

    def send_to_server(self, message):
        self.connect()
        self.response = None

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

        return self.response
