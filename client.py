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
import queue


class MLP(nn.Module):
    def __init__(self,inputSize, width, layersN):
        super(MLP, self).__init__()
        self.layersN = layersN
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inputSize, width))
        for _ in range(layersN-1):
            self.layers.append(nn.Linear(width, width))
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x
    def get_last_layer_input(self,x):
        with torch.no_grad():
            for i in range(self.layersN-1):
                x = torch.relu(self.layers[i](x))
        return x


class modelItem:
    def __init__(self,shape,lr=0.0001):
        self.shape = shape
        self.lr = lr
        self.trainN = 0
        self.testN = 0
        self.model = None
        self.build_model()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr)
    def build_model(self):
        I,W,L = self.shape
        if L < 1:
            L = 1
        self.model = MLP(I,W,L)


class client:
    def __init__(self, server_ip, server_port, Tx, prefer=1, max_send=5, batchsize=128, zipMetric=None, send_pkg_CD=0.05):
        self.server_ip = server_ip
        self.server_port = server_port
        self.prefer = prefer

        # the maximum # of batch in sending buffer (used in wait_complete())
        self.max_send_N = max_send
        # set the size of receive buffer to be twice of sending buffer
        self.max_receive_N = max_send*2

        self.Tx = Tx
        self.socket = None
        self.modelItem = None
        self.model = None
        self.batchQ = queue.Queue(maxsize=max_send)
        self.gradient = queue.Queue(maxsize=self.max_receive_N)
        self.send_pkg_CD = send_pkg_CD
        self.running = True
        self.state = "offline"
        self.trainset_loader = None
        self.testset_loader = None
        self.itertrain = None
        self.itertest = None
        self.SDL = {}
        self.batchSize = batchsize
        self.zipMetric = zipMetric
        self.mode = "train"
        self.testLock = threading.Lock()
        self.trainLock = threading.Lock()

        
    
    def info_to_json(self):
        return json.dumps({
            "type": "info",
            "Tx": self.Tx,
            "prefer": self.prefer,
            "Buffer Size": self.max_send_N
        })

    # TODO generate message (type, mes)
    # TODO 多线程处理发送和接收。目前发送需要上一轮接收完毕。可以考虑每一段时间接收一次  Final: 还是单线程，但是用了另一种方法来完成
    def generate_message(self, type, mes=None):
        if type == "End":
            m = json.dumps({"type": type})
            print(m)
            return m
        return json.dumps({"type": type})
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
                    if self.batchQ.full(): #if there is too much data waiting gradient, stop sending
                        #print("too large now")
                        self.wait_complete()
                        continue    
                    batch = self.get_next_train_batch()
                    if batch is None:
                        # if there is still data that has not received gradient, continue receiving.
                        self.wait_complete()
                        self.mode = "test"
                        continue 

                if self.mode == "test":
                    batch = self.get_next_test_batch()
                    if batch is None:
                        self.formal_end() 

                if batch is not None:
                    self.compute_smashed_data(batch)

            if self.state == "ready upload":
                self.upload_SDL_to_server()
            if self.state == "wait gradient":
                self.receive_gradient_from_server(self.send_pkg_CD)  #Note to add timeout
            if self.state == "update local model":
                self.update_model()
        print("========= Client stopped =========")

    def formal_end(self):
        print("end")
        m = self.generate_message("End")
        self.sendmes(m)
        time.sleep(3)
        self.disconnect()
    def prepare_data(self):
    

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        if not os.path.exists('./data/MNIST'):
            print("-> prepare data: Downloading MNIST dataset...")
        try:

            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batchSize, shuffle=True, drop_last=False)
            self.trainset_loader = trainloader
            self.itertrain = iter(trainloader)


            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.batchSize, shuffle=False, drop_last=False)
            self.testset_loader = testloader
            self.itertest = iter(testloader)

        except Exception as e:
            print(f" !! State: {self.state} | prepare_data() error: {e}")
            self.disconnect()
            return
        print(f" -> prepare data complete | trainset len: {len(trainset)} | testset len: {len(testset)}")
    
    def get_next_train_batch(self):
        try:
            with self.trainLock:
                return next(self.itertrain)
        except StopIteration:
            return None
    def get_next_test_batch(self):
        try:
            with self.testLock:
                return next(self.itertest)
        except StopIteration:
            return None
    def upload_SDL_to_server(self):
        #print(f"==> Start: upload_SDL_to_server()")
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
        if self.mode == "train":
            self.state = "wait gradient"
        if self.mode == "test":
            self.state = "working"
        self.SDL = None
        #print("==> Complete: upload_SDL_to_server()")
        #print("-------------------------------------------")
    def receive_gradient_from_server(self, timeout=None):
        #print(f"==> Start: receive_gradient_from_server()")
        if self.socket is None:
            print(f" !! State: {self.state} | receive_gradient_from_server() -> error: socket is None")
            self.disconnect()
            return
        try:
            self.socket.settimeout(timeout)
            msg = self.socket.recv(pow(2,30)).decode()
            data = json.loads(msg)
        except socket.timeout:
            self.state = "working"
            return                        #Note add except socket.timeout
        except Exception as e:
            print(f"!! State: {self.state} | receive_gradient_from_server() -> error: {e}")
            self.disconnect()
            return
        type = data.get("type")
        if type == "hb":
            return self.receive_gradient_from_server(timeout)
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
        
        # split the gradient into multiple batches
        split = torch.split(gradient, self.batchSize, dim=0)
        for g in split:
            self.gradient.put(g)
        self.state = "update local model"
        
        #print(f"==> complete: receive_gradient_from_server()")
        #print("-------------------------------------------")
        
        
    def update_model(self):
        #print(f"==> Start: update_model()")
        if self.model is None:
            print(f" !! State: {self.state} | update_model() -> error: model is None")
            self.disconnect()
            return
        if self.gradient.empty():
            print(f" ~ State: {self.state} | update_model() -> return with empty gradient")
            return
        if self.batchQ.empty():
            print(f" ~ State: {self.state} | update_model() -> return with empty batchQ")
            return
        # receive n sample's gradient, retrive the samples
        try:
            one_batch = self.batchQ.get_nowait()
            batch_data, label = one_batch
        except queue.Empty:
            print("batchQ is empty")
            self.disconnect()
            return
        try:
            gradient = self.gradient.get_nowait()
        except queue.Empty:
            print("self.gradient is empty")
            self.disconnect()
            return
        # get the input data at Lth layer

        batch_data = batch_data.view(batch_data.size(0), -1)
        L_input = self.model.get_last_layer_input(batch_data)
        # receive L+1th layer gradient, backward to Lth layer
        layer_L = self.model.layers[-1]
        L_weight_gradient = torch.matmul(L_input.T, gradient).T
        L_bias_gradient = gradient.sum(dim=0)

        layer_L.weight.grad = L_weight_gradient.detach().clone()
        layer_L.bias.grad = L_bias_gradient.detach().clone()

        optimizer = self.modelItem.optimizer 
        optimizer.step()
        optimizer.zero_grad()
        self.state = "working"
        #print(f"==> complete: update_model()")
        #print("-------------------------------------------")

    def wait_complete(self):
        # first receive all gradient
        batchN = self.batchQ.qsize()
        if batchN == 0:
            return
        while self.gradient.qsize() < batchN:
            self.receive_gradient_from_server(0.5)
        #print(batchN)
        #print(self.gradient.qsize())
        
        # update all
        while not self.batchQ.empty():
            self.update_model()

    def compute_smashed_data(self, batch):
        #print(f"==> Start: compute_smashed_data()")
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

        if batch is None:
            print(f" !! State: {self.state} | compute_smashed_data() -> error: batch is None")

        self.model.eval()
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
        if self.mode == "train":   #note here
            self.batchQ.put(batch)
        self.state = "ready upload"
        #print(f"==> Complete: compute_smashed_data()")
        #print("-------------------------------------------")

    def upload_info(self):
        print(f"==> Start: upload_info()")
        self.sendmes(self.info_to_json())
        self.state = "online waiting"
        print(f"==> Complete: upload_info()")
        print("-------------------------------------------")
    def try_login(self):
        print(f"==> Start: try_login()")
        try:
            message = self.socket.recv(1024).decode('utf8')
        except Exception as e:
            print(f" !! State: {self.state} | try_login() -> error: {e}")
            self.disconnect()
            return
        try:
            parsed_message = json.loads(message)
        except json.JSONDecodeError:
            print(f" !! State: {self.state} | try_login() decode json -> MSG: {1}, error: {e}")
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
            data = self.socket.recv(pow(2,30)).decode('utf8')
        except socket.error as e:
            print(f" !! State: {self.state} | receive_model_from_server() recv msg error -> MSG: {1}, error: {e}")
            self.disconnect()
            return
        try:
            parsed_message = json.loads(data)
        except json.JSONDecodeError as e:
            print(f" !! State: {self.state} | receive_model_from_server() json decode error -> MSG: {1}, error: {e}")
            self.disconnect()
            return
        
        msg_type = parsed_message.get("type", "")
        if msg_type == "hb":
            return self.receive_model_from_server()
        
        if msg_type != "model":
            print(f" !! State: {self.state} | receive_model_from_server() -> receive type \"{msg_type}\", expect type \"model\"")
            self.disconnect()
            return
        
        lr = parsed_message.get("lr")
        shape = parsed_message.get("shape")
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
        
        self.modelItem = modelItem(shape,lr)
        self.model = self.modelItem.model
        if self.model is None:
            print(f" !! State: {self.state} | receive_model_from_server() -> failed to create MLP with [I W O L] = [{shape}]")
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
    client = client("127.0.0.1", 65432,123,prefer=1)
    print('press enter to start client')
    input()
    client.start()