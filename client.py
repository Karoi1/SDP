import socket
import json
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64
import torchvision
import torchvision.transforms as transforms
import os
import io

class client:
    def __init__(self, server_ip, server_port, Tx, prefer=1, batchN=64, zipMetric=None):
        self.server_ip = server_ip
        self.server_port = server_port
        self.prefer = prefer
        self.Tx = Tx
        self.socket = None
        self.model = None
        self.batch = None
        self.gradient = None
        self.running = True
        self.state = "offline"
        self.trainset_loader = None
        self.testset_loader = None
        self.SDL = {}
        self.batchN = batchN
        self.batch_idx = 0
        self.zipMetric = zipMetric
        self.mode = "train"
    
    def info_to_json(self):
        return json.dumps({
            "type": "info",
            "Tx": self.Tx,
            "prefer": self.prefer
        })

    
    def start(self):
        self.prepare_data()
        print("========= Client started =========")
        while self.running:
            if self.state == "offline":
                self.connect()
            if self.state == "queueing":
                self.try_login()

            if self.state == "online":
                self.upload_info()

            if self.state == "online waiting":
                self.receive_model_from_server()
            if self.state == "working":
                if self.mode == "train":
                    self.batch = self.get_next_train_batch()
                if self.mode == "test":
                    self.batch = self.get_next_test_batch()
                if self.batch is not None:
                    self.compute_smashed_data(self.batch)
                if self.batch is None and self.mode == "train":
                    self.mode = "test"
                if self.batch is None and self.mode == "test":
                    self.disconnect()
            if self.state == "ready upload":
                self.upload_SDL_to_server()
            if self.state == "wait gradient":
                self.receive_gradient_from_server()
            if self.state == "update local model":
                self.update_model()
        print("========= Client stopped =========")


    def prepare_data(self):
    
        # 定义数据预处理操作
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
            transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
        ])
        if not os.path.exists('./data/MNIST'):
            print("-> prepare data: Downloading MNIST dataset...")
        try:
            # 下载并加载训练集
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
            self.trainset_loader = trainloader

            # 下载并加载测试集
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
            self.testset_loader = testloader

        except Exception as e:
            print(f" !! State: {self.state} | prepare_data() error: {e}")
            self.disconnect()
            return
        print(f" -> prepare data complete | trainset len: {len(trainset)} | testset len: {len(testset)}")
    
    def get_next_train_batch(self):
        try:
            return next(iter(self.trainset_loader))
        except StopIteration:
            return None
    def get_next_test_batch(self):
        try:
            return next(iter(self.testset_loader))
        except StopIteration:
            return None
    def upload_SDL_to_server(self):
        print(f"==> Start: upload_SDL_to_server()")
        if not self.SDL:
            print(f" !! State: {self.state} | upload_SDL_to_server() -> error: SDL is empty")
            self.disconnect()
            return
        try:
            sdl_json = json.dumps(self.SDL)
        except Exception as e:
            print(f"!! State: {self.state} | upload_SDL_to_server() -> encode json error: {e}")
            self.disconnect()
            return
        self.sendmes(sdl_json)
        self.state = "wait gradient"
        self.SDL = None
        print("==> Complete: upload_SDL_to_server()")
        print("-------------------------------------------")
    def receive_gradient_from_server(self):
        print(f"==> Start: receive_gradient_from_server()")
        if self.socket is None:
            print(f" !! State: {self.state} | receive_gradient_from_server() -> error: socket is None")
            self.disconnect()
            return
        try:
            msg = self.socket.recv(1024000).decode()
            data = json.loads(msg)
        except Exception as e:
            print(f"!! State: {self.state} | receive_gradient_from_server() -> error: {e}")
            self.disconnect()
            return
        type = data.get("type")
        if type == "hb":
            return self.receive_gradient_from_server()
        if type != "gradient":
            print(f" !! State: {self.state} | receive_gradient_from_server() -> error: type is not gradient")
            self.disconnect()
            return
        
        gradient = data.get("G")
        if gradient is None:
            print(f" !! State: {self.state} | receive_gradient_from_server() -> error: gradient is None")
            self.disconnect()
            return
        try:
            gradient = torch.load(io.BytesIO(base64.b64decode(gradient)),weights_only=True)
        except Exception as e:
            print(f" !! State: {self.state} | receive_gradient_from_server() -> error decoding gradient: {e}")
            self.disconnect()
            return
        
        self.state = "update local model"
        self.gradient = gradient
        print(f"==> complete: receive_gradient_from_server()")
        print("-------------------------------------------")
        
        
    def update_model(self):
        print(f"==> Start: update_model()")
        if self.model is None:
            print(f" !! State: {self.state} | update_model() -> error: model is None")
            self.disconnect()
            return
        if self.gradient is None:
            print(f" !! State: {self.state} | update_model() -> error: gradient is None")
            self.disconnect()
            return
        # receive L+1th layer gradient, backward to Lth layer
        layer_L = self.model.layers[-1]
        L_weight_gradient = torch.matmul(self.gradient, layer_L.weight)
        L_bias_gradient = self.gradient.sum(dim=0)
        if layer_L.weight.grad is None:
            layer_L.weight.grad = L_weight_gradient.detach().clone()
        else:
            layer_L.weight.grad += L_weight_gradient.detach().clone()

        if layer_L.bias.grad is None:
            layer_L.bias.grad = L_bias_gradient.detach().clone()
        else:
            layer_L.bias.grad += L_bias_gradient.detach().clone()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)  # TODO: 用消息让server和client的lr同步
        optimizer.step()
        optimizer.zero_grad()

        self.gradient = None
        self.state = "online waiting"
        print(f"==> complete: update_model()")
        print("-------------------------------------------")
    def create_dynamicMLP(self, inputSize, width, layersN):
        if layersN < 1:
            print(f" !! State: {self.state} | create_dynamicMLP() -> error: layers must be at least 1")
            self.disconnect()
            return None
        class MLP(nn.Module):
            def __init__(self,inputSize, width, layersN):
                super(MLP, self).__init__()
                self.layers = nn.ModuleList()
                self.layers.append(nn.Linear(inputSize, width))
                for _ in range(layersN-1):
                    self.layers.append(nn.Linear(width, width))
            
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        return MLP(inputSize, width, layersN)


    def compute_smashed_data(self, batch):
        print(f"==> Start: compute_smashed_data()")
        if self.trainset_loader is None:
            print(f" !! State: {self.state} | compute_smashed_data() -> error: train set is None")
            self.disconnect()
            return
        if self.testset_loader is None:
            print(f" !! State: {self.state} | compute_smashed_data() -> error: test set is None")
            self.disconnect()
            return
        if self.model is None:
            print(f" !! State: {self.state} | compute_smashed_data() -> error: model is None")
            self.disconnect()
            return

        self.model.eval()
        if batch is None:
            print(f" !! State: {self.state} | compute_smashed_data() -> error: batch is None")
        
        batch_data, batch_labels = batch
        # flatten the data
        batch_data = batch_data.view(batch_data.size(0), -1)
        with torch.no_grad():
            smashed_data = self.model.forward(batch_data)
        
        self.SDL = {
            "type": f"{self.mode} SDL",
            "SD": smashed_data.tolist(),
            "L": batch_labels.tolist(),
            "batchN": len(batch_labels)
        }
        self.state = "ready upload"
        print(f"==> Complete: compute_smashed_data()")
        print("-------------------------------------------")

    def upload_info(self):
        print(f"==> Start: upload_info()")
        self.sendmes(self.info_to_json())
        self.state = "online waiting"
        print(f"==> Complete: upload_info()")
        print("-------------------------------------------")
    def try_login(self):
        print(f"==> Start: try_login()")
        try:
            message = self.socket.recv(1024)
        except Exception as e:
            print(f" !! State: {self.state} | try_login() -> error: {e}")
            self.disconnect()
            return
        try:
            parsed_message = json.loads(message)
        except json.JSONDecodeError:
            print(f" !! State: {self.state} | try_login() decode json -> MSG: {message}, error: {e}")
            self.disconnect()
            return

        print(f" -> try_login(): receive server reply: {parsed_message}")

        msg_type = parsed_message.get("type", "")
        if msg_type == "hb":
            return self.try_login()
        if self.state not in ["offline", "queueing"]:
            print(f" !! State: {self.state} | try_login(): receive type: \"{msg_type}\", expect type: \"loginState\"")
            return
        
        if self.state in ["offline", "queueing"] and msg_type == "loginState":
            value = parsed_message.get("value", "")
            if value == "Full":
                self.state = "offline"
                self.disconnect()
                print(" -- Server full, quitting")
            elif value == "Queueing":
                self.state = "queueing"
                print(" == Queueing")
            elif value == "OK":
                self.state = "online"
                print(" -> Successful login")        
        print(f"==> Complete: try_login()")
    def connect(self):
        print(f"==> Start: connect()")
        if self.socket is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.socket.connect((self.server_ip, self.server_port))
                self.state = "queueing"
            except socket.error as e:
                print(f" !! State: {self.state} | connect() -> error: {e}")
                self.disconnect()
                return
            print(f"==> Complete: connect()")
            print("-------------------------------------------")
    def disconnect(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        self.running = False
        self.state = "offline"
        print("==> Disconnected")

    def sendmes(self, mes):
        try:
            self.socket.sendall(mes.encode('utf8'))
        except socket.error as e:
            print(f" !! State: {self.state} | sendmes -> MSG: {mes}, error: {e}")
            self.disconnect()
            return

    def receive_model_from_server(self):
        print(f"==> Start: receive_model_from_server()")
        try:
            data = self.socket.recv(1024000).decode('utf8')
        except socket.error as e:
            print(f" !! State: {self.state} | receive_model_from_server() recv msg error -> MSG: {data}, error: {e}")
            self.disconnect()
            return
        try:
            parsed_message = json.loads(data)
        except json.JSONDecodeError as e:
            print(f" !! State: {self.state} | receive_model_from_server() json decode error -> MSG: {data}, error: {e}")
            self.disconnect()
            return
        
        msg_type = parsed_message.get("type", "")
        if msg_type == "hb":
            return self.receive_model_from_server()
        
        if msg_type != "model":
            print(f" !! State: {self.state} | receive_model_from_server() -> receive type \"{msg_type}\", expect type \"model\"")
            self.disconnect()
            return
        
        shape = parsed_message.get("shape")
        I, W, L = shape
        encoded_weight = parsed_message.get("binary_data")

        if not encoded_weight:
            print(f" !! State: {self.state} | receive_model_from_server() -> model weight is empty")
            self.disconnect()
            return
        
        try:
            model_weight = base64.b64decode(encoded_weight)
        except Exception as e:
            print(f" !! State: {self.state} | receive_model_from_server() -> error decoding model weights: {e}")
            self.disconnect()
            return
        
        self.model = self.create_dynamicMLP(I, W, L)
        if self.model is None:
            print(f" !! State: {self.state} | receive_model_from_server() -> failed to create MLP with [I W O L] = [{I} {W} {L}]")
            self.disconnect()
            return
        
        try:
            self.model.load_state_dict(torch.load(io.BytesIO(model_weight), weights_only=True))
            self.model.eval()
        except Exception as e:
            print(f" !! State: {self.state} | receive_model_from_server() -> error loading model weights: {e}")
            self.disconnect()
            return
        self.state = "working"
        print("==> Complete: receive_model_from_server()")
        print("-------------------------------------------")


if __name__ == "__main__":
    client = client("127.0.0.1", 65432,123)
    print('press enter to start client')
    input()
    client.start()