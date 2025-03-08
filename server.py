import socket
import concurrent.futures
import threading
import signal
import sys
import time
import queue
import json
import torch
import torch.nn as nn
import base64
import io
import random
import numpy as np

class modelManager:
    def __init__(self):
        self.idcounter = 0
        self.Listmodel = np.array([], dtype=modelItem)
    def addModel(self, shape, split_layer):
        """
        Create a new model and store in self.Listmodel. ID is automatically generated
        Parameters:
            :shape: shape of model stored in list. Format: [inputSize, Width, outputSize, #layers]
            :split_layer: specify the split layer of the model.
            :Where server-side compute layer = [split_layer:]
        Return:
            :Bool: Whether model is successful created
            :Str: error message
        """
        
        if len(shape) != 4:
            return False, f" !! Model Manager | addModel(): expect len(shape)==4, get {len(shape)}"
        I,W,O,L = shape
        if I is None or W is None or O is None or L is None:
            return False, f" !! Model Manager | addModel(): One of the item in shape is None"
        self.idcounter += 1
        modelID = self.idcounter
        Newmodel = modelItem(shape, modelID, split_layer)
        self.Listmodel = np.append(self.Listmodel, Newmodel)
        return True, ""

    def getRandomModel(self, split_layer):
        """
        Get a random Full model which is of the corresponding split layer
        Parameters:
            :id: the id of the model
        Return:
            :modelItem(): if the model is found
            :None: if no model found
        """
        L = np.array([item for item in self.Listmodel if item.split_layer==split_layer])
        if L.size == 0:
            return None
        return random.choice(L)

    def getModel(self, id):
        """
        Get the Full model of the corresponding id
        Parameters:
            :id: the id of the model
        Return:
            :modelItem(): if the model is found
            :None: if no model found
            :Str: error message
        """
        model = next((item for item in self.Listmodel if item.id == id), None)
        if model is None:
            return None, f" !! Model Manager | getModel(): no model of {id} found"
        return model, ""
    def getClientModel(self, id):
        """
        Get the client-side model of the corresponding id
        Parameters:
            :id: the id of the model
        Return:
            :modelItem(): if the model is found
            :None: if no model found
            :Str: error message
        """
        full, error = self.getModel(id)
        if full is None:
            return None, error
        if full.split_layer < 2:
            return None, f" !! Model Manager | getClientModel: model.split_layer = {full.split_layer} < 2"
        
        I,W,O,L = full.shape
        split_layer = full.split_layer
        subShape = np.array([I,W,W,split_layer-1])
        CM = modelItem(subShape)

        # copy the full model weight into sub model
        for i in range(split_layer):
            print(CM.layers[i].weight.shape)
            CM.layers[i].weight.data.copy_(full.layers[i].weight.data)
            CM.layers[i].bias.data.copy_(full.layers[i].bias.data)
        return CM, ""
    
    def forward_for_id(self, id, smashed_data):
        """
        Forward the smashed data(at split layer) for the model of corresponding id. Only eval, will not train the model
        Parameters:
            :id: the id of the model
            :smashed_data: smashed data at split layer
        Return:
            :Tensor: if the model is found, forward and output the smashed data
            :None: if no model found
            :Str: error message
        """
        item, error = self.getModel(id)
        if item is None:
            return None, error
        model = item.model
        split_layer = item.split_layer
        x = model.forward(smashed_data, split_layer)
        return x, ""
    
    def train_for_id(self, id, smashed_data):
        """
        Train the model of corresponding id with smashed data(at split layer). 
        Parameters:
            :id: the id of the model
            :smashed_data: smashed data at split layer
        Return:
            :Tensor: if the model is found, forward and output the smashed data
            :None: if no model found
            :Str: error message
        """
        item, error = self.getModel(id)
        if item is None:
            return None, error
        model = item.model
        optimizer = item.optimizer
        lock = item.lock

        with lock:
            smash_data = torch.tensor(smashed_data, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            criterion = nn.CrossEntropyLoss()

            outputs = model.forward(smash_data)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self.get_split_layer_gradient(id)

    def get_split_layer_gradient(self, id):
        """
        get the gradient at split layer for the model of corresponding id.
        Parameters:
            :id: the id of the model
        Return:
            :Tensor: if the model is found, forward and output the smashed data
            :None: if no model found
            :Str: error message
        """
        item, error = self.getModel(id)
        if item is None:
            return None, error
        model = item.model

        gradients = []
        i = 0
        for param in model.parameters():
            #print(param.shape)
            if param.grad is not None:
                #print(i)
                gradients.append(param.grad.data.clone())
                i+=1
                if i >= 2:
                    break
        return gradients[0], ""

class modelItem:
    def __init__(self, shape, id=-1, lr=0.001, split_layer=0):
        self.id = id
        self.split_layer = split_layer
        self.shape = shape
        self.model = None
        self.lr = lr
        self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.lock = threading.Lock()
    def build_model(self):
        I,W,O,L = self.shape
        self.model = DynamicMLP(I,W,O,L)
        

    

class DynamicMLP(nn.Module):
    def __init__(self, I, W, O, L):
        super(DynamicMLP, self).__init__()
        # 创建输入层
        self.layers = nn.ModuleList([nn.Linear(I, W)])
        
        # 创建隐藏层
        for _ in range(L - 2):  # 减去输入层和输出层
            self.layers.append(nn.Linear(W, W))
        
        # 创建输出层
        if L > 1:
            self.layers.append(nn.Linear(W, O))

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x, start_layer=0):
        """
        从self.start_layer层开始处理输入 x，直到输出。
        :param x: 输入张量
        :return: 输出张量
        """
        a = 0
        if start_layer < 0 or start_layer >= len(self.layers):
            raise ValueError(f"Invalid layer index start_layer={start_layer}. Must be between 0 and {len(self.layers)-1}.")

        # 从指定的层开始处理输入 x
        for i in range(start_layer, len(self.layers) - 1):
            a+=1
            x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)  # 最后一层不使用 ReLU 激活函数
        return x


class Client:
    def __init__(self, ip_address, tx=None, socket=None, state="queueing", prefer=1,subscribe="gradient"):
        self.ip_address = ip_address
        self.Tx = tx
        self.socket = socket
        self.state = state
        self.prefer = prefer
        self.SD = None
        self.L = None
        self.gradient = None
        self.batchN = None
        self.subscribe = subscribe
        self.modelID = None

class Server:
    def __init__(self, host, port, max_connections, max_workers, max_wait_queue=5, broadcast_interval=60, input_size=784, output_size=10, modelL=3, modelW=5):
        
        # ip, port, listen socket
        self.host = host
        self.port = port
        self.server_socket = None

        # max in accept()
        self.max_connections = max_connections
        # max connections
        self.max_workers = max_workers
        # current connections
        self.current_worker = 0
        # boardcast interval
        self.broadcast_interval = broadcast_interval  
        # queue length
        self.max_wait_queue = max_wait_queue
        # queue for waiting clients
        self.wait_queue = queue.Queue(maxsize=max_wait_queue)

        # lock for workersN
        self.worker_lock = threading.Lock()
        # lock for client list
        self.clients_lock = threading.Lock()
        # lock for model
        self.Listmodel_lock = threading.Lock()

        # thread pool for receiving messages
        self.receiverpool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        # thread pool for sending messages
        self.senderpool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # list of all connected clients
        self.clients = []
        self.running = True

        # model_distributor CD
        self.DMCD = 5
        self.UDCD = 5

        self.inputSize = input_size
        self.outputSize = output_size
        self.modelL = modelL
        self.modelW = modelW
        self.Listmodel = {}
        self.model_id_counter = 0
        self.lr = 0.001
    def start(self):
        print("creating socket...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_connections)

        print("initialize brocast thread...")
        # 启动定时广播线程
        broadcast_thread = threading.Thread(target=self.background_broadcast)
        broadcast_thread.daemon = True  # 设置为守护线程，随主线程退出而退出
        #broadcast_thread.start()

        #print("initialize checker thread...")
        # check queue thread
        check_queue_thread = threading.Thread(target=self.check_wait_queue)
        check_queue_thread.daemon = True  # 设置为守护线程，随主线程退出而退出
        check_queue_thread.start()

        # check client thread
        check_client_thread = threading.Thread(target=self.check_valid_clients)
        check_client_thread.daemon = True  # 设置为守护线程，随主线程退出而退出
        check_client_thread.start()

        model_distributor_thread = threading.Thread(target=self.model_distributor)
        model_distributor_thread.daemon = True
        model_distributor_thread.start()

        model_updater_thread = threading.Thread(target=self.update_model)
        model_updater_thread.daemon = True
        model_updater_thread.start()

        gradient_distributor_thread = threading.Thread(target=self.gradient_distributor)
        gradient_distributor_thread.daemon = True
        gradient_distributor_thread.start()

        print("========********[ Server On Board ]********========")
        print(f" === Listen at {self.host}: {self.port}")

        try:
            while self.running:
                client_socket, client_addr = self.server_socket.accept()
                print(f"-> Listen: connection from: {client_addr}")

                client = Client(client_addr, socket=client_socket)
                with self.clients_lock:
                    self.clients.append(client)
                
                with self.worker_lock:
                    if self.current_worker < self.max_workers:
                        m = self.generate_messages("loginState", "OK")
                        self.send_client_mes(client, m)
                        self.receiverpool.submit(self.listen_client_mes, client)
                        client.state = "online"
                        print(f"-> Listen: {client_addr} direct to listen_client_mes()")
                    else:
                        # 线程池满，处理队列或关闭连接
                        if self.wait_queue.full():
                            m = self.generate_messages("loginState", "FULL")
                            self.send_client_mes(client, m)
                            client.socket.close()
                            print(f"-> Listen: Queue Full, connection break: {client_addr}")
                        else:
                            client.state = "queueing"
                            m = self.generate_messages("loginState", "Queueing")
                            self.send_client_mes(client, m)
                            self.wait_queue.put(client)
                            print(f"-> Listen: {client_addr} direct to queue")


        except KeyboardInterrupt:
            print("Server stopped.")
        finally:
            self.running = False
            self.server_socket.close()
            self.receiverpool.shutdown()
            self.senderpool.shutdown()
            broadcast_thread.join()
            check_queue_thread.join()
    
    def create_DynamicMLP(self, I, W, O, L, start_layer=1, sub=False):
        if not sub:
            self.model_id_counter += 1
            return DynamicMLP(I, W, O, L, self.model_id_counter, start_layer)
        return DynamicMLP(I, W, O, L, -1, start_layer)
    def back_propagate(self, model, optimizer, lock, smashed_data, labels):
        with lock:
            try:
                smash_data = torch.tensor(smashed_data, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.long)

                criterion = nn.CrossEntropyLoss()

                outputs = model.forward(smash_data)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                return self.get_Nth_layer_gradients(model)
            except Exception as e:
                print(f" !! back_propagate error | {e}")
    def foward_propagate(self, model, smashed_data):
        return model.forward(smashed_data)
    def get_Nth_layer_gradients(self, model):
        gradients = []
        i = 0
        for param in model.parameters():
            #print(param.shape)
            if param.grad is not None:
                #print(i)
                gradients.append(param.grad.data.clone())
                i+=1
                if i >= 2:
                    break
        return gradients[0]

    def generate_messages(self, type, message=None):
        if type == "loginState":
            return json.dumps({"type": type, "value": message})
        if type == "model":
            binary_data = self.tensor_to_byte(message[0].state_dict())
            shape = message[1:]
            return json.dumps({"type": type, "shape": shape, "binary_data": binary_data})
        if type == "hb":
            return json.dumps({"type": type}).encode("utf-8")
        if type == "gradient":
            binary_data = self.tensor_to_byte(message)
            return json.dumps({"type": type, "G": binary_data})
        
        print(" !! generate_messages | invalid type: \"{type}\", m: \"{message}\"")

    def check_valid_clients(self):
        """定期检查客户端连接状态并清理离线客户端"""
        while self.running:
            time.sleep(10)  # 每隔10秒检查一次

            # 获取离线客户端
            offline_clients = []
            m = self.generate_messages("hb")
            with self.clients_lock:
                for client in self.clients:
                    try:
                        # 尝试从客户端发送一个心跳请求
                        client.socket.sendall(m)
                        # 如果客户端断开，会抛出异常
                    except (socket.error, OSError):
                        if client.SD is None and client.L is None and client.gradient is None:
                            offline_clients.append(client)

                # 从客户端列表中移除离线客户端
                self.clients = [c for c in self.clients if c not in offline_clients]

            # 关闭离线客户端的socket
            for client in offline_clients:
                try:
                    client.socket.close()
                except Exception:
                    pass
            if len(offline_clients) != 0:
                print(f"-> Cleaner: Removed {len(offline_clients)} offline client")


    def check_wait_queue(self):
        """定期检查等待队列并调度任务"""
        while self.running:
            time.sleep(1)  # 每隔1秒检查一次
                # 如果线程池有空闲资源，并且等待队列不为空
            with self.worker_lock:
                    if self.current_worker < self.max_workers:
                        try:
                            client = self.wait_queue.get_nowait()
                            m = self.generate_messages("loginState", "OK")
                            self.send_client_mes(client, m)
                            self.receiverpool.submit(self.listen_client_mes, client)
                            client.state = "online waiting"
                            print(f"-> Queue: direct {client.ip_address} to listen_client_mes()")
                            
                        except queue.Empty:
                            pass
                        except Exception as e:
                            print(f" !! check_wait_queue() Error: {e}")
                            
    
    def send_client_mes(self, client, mes):
        try:
            client.socket.sendall(mes.encode('utf-8'))
        except Exception as e:
            print(f" !! send_client_mes Error | ip: {client.ip_address} | e: {e} | m: {mes}")
        
    def divide_model(self, client, model):
        sub_model = self.create_DynamicMLP(self.inputSize, self.modelW, self.modelW, client.prefer, sub=True)
        for i in range(client.prefer):
            print(sub_model.layers[i].weight.shape)
            sub_model.layers[i].weight.data.copy_(model.layers[i].weight.data)
            sub_model.layers[i].bias.data.copy_(model.layers[i].bias.data)

        return sub_model, self.inputSize, self.modelW, client.prefer

    def model_distributor(self):
        while self.running:
            time.sleep(self.DMCD)
            with self.clients_lock:
                counter = 0
                for client in self.clients:
                    if client.state == "online waiting":
                        model, _, lock = self.select_model(client)
                        if model is None:
                            print("no model found")
                            print(client.modelID)
                            break
                        with lock:
                            sub_model,I,W,L = self.divide_model(client,model)
                        m = self.generate_messages("model", [sub_model,I,W,L])
                        self.send_client_mes(client, m)
                        client.state = "working"
                        counter += 1
            if counter != 0:
                print(f"-> Model distributor: send model to {counter} clients")

    def select_model(self, client, Rand=False):
        prefer = client.prefer
        if prefer not in self.Listmodel.keys():
            print(f" !! select_model | -> Client: {client.ip_address} prefer: {prefer}, no model found")
            return
        if Rand:
            model, optimizer, lock = random.choice(self.Listmodel[prefer])
        if not Rand:
            for k in self.Listmodel.keys():
                model, optimizer, lock = next(((m,o,l) for m,o,l in self.Listmodel[k] if m.id == client.modelID), (None,None,None))
                if model is not None:
                    break
        return model, optimizer, lock

    def add_model(self, start_layer, new=False):
        if start_layer not in self.Listmodel.keys():
            Newmodel = self.create_DynamicMLP(self.inputSize, self.modelW, self.outputSize, self.modelL, start_layer)
            optimizer = torch.optim.Adam(Newmodel.parameters(),lr=self.lr)
            lock = threading.Lock()
            self.Listmodel[start_layer] = [(Newmodel, optimizer, lock)]
        if start_layer in self.Listmodel.keys() and new:
            Newmodel = self.create_DynamicMLP(self.inputSize, self.modelW, self.outputSize, self.modelL, start_layer)
            optimizer = torch.optim.Adam(Newmodel.parameters(),lr=self.lr)
            lock = threading.Lock()
            self.Listmodel[start_layer].append((Newmodel, optimizer, lock))
        return self.model_id_counter

    def receive_clientTx(self, client, tx):
        client.tx = tx
        client.state = "ready"
        print(f"客户端 {client.ip_address} Tx: {tx}")
    
    def tensor_to_byte(self, tensor):
        binary_data = io.BytesIO()
        torch.save(tensor, binary_data)
        return base64.b64encode(binary_data.getvalue()).decode("utf-8")
    def update_model(self):
        while self.running:
            time.sleep(self.UDCD)
            with self.clients_lock:
                #print("update model check")
                for client in self.clients:
                    #print(f"ip: {client.ip_address}")
                    if client.SD is not None and client.L is not None:
                        smashed_data = client.SD
                        labels = client.L
                        model, optimizer, lock = self.select_model(client)
                        gradient = self.back_propagate(model, optimizer, lock, smashed_data, labels)
                        client.gradient = gradient
                        #m = self.generate_messages("gradients", gradients)
                        #self.send_client_mes(client, m)
                        client.SD = None
                        client.L = None
                        #print(f"-> Client: {client.ip_address} gradients: {gradients.shape}")

    def gradient_distributor(self):
        while self.running:
            time.sleep(5)
            with self.clients_lock:
                counter = 0
                for client in self.clients:
                    if client.gradient is not None:
                        m = self.generate_messages("gradient", client.gradient)
                        self.send_client_mes(client, m)
                        client.gradient = None
                        counter += 1
            if counter!= 0:
                print(f"-> Gradient Distributor: sent gradient to {counter} clients")

    def parse_handle_mes(self, client, data):
        #print(f"From {client.ip_address}: {data}")
        data = json.loads(data)
        type = data.get('type')
        if type == "info":
            client.Tx = data.get('Tx')
            prefer = data.get('prefer')
            client.prefer = prefer
            newModelID = self.add_model(client.prefer)
            client.modelID = newModelID

            if client.state == "online":
                client.state = "online waiting"
            return
        if type == "train SDL":
            if client.state == "working":
                client.SD = data.get('SD')
                client.L = data.get('L')
                client.batchN = data.get('batchN')
                print(f"received batch {len(client.L)}")
            else:
                print(f"~ parse_handle_mes: receive train SDL for non working client | state: {client.state}, ip: {client.ip_address}")
            return
        if type == "test SDL":
            SD = data.get('SD')
            L = data.get('L')
            batchN = data.get('batchN')
            return
        print(f"~ parser: unknown data | ip: {client.ip_address} | d: {data}")

    def listen_client_mes(self, client):
        try:
            with self.worker_lock:
                self.current_worker += 1
                print(f"+++ current workers: {self.current_worker}")

            while self.running:
                try:
                    mes = client.socket.recv(1024000).decode('utf-8')
                    if not mes:
                        break
                    self.parse_handle_mes(client, mes)
                except socket.timeout:
                    print(f"~ listen_client_mes TimeOut | ip: {client.ip_address}")
                    break
                except socket.error as e:
                    print(f"~ listen_client_mes Error | ip: {client.ip_address} | e: {e}")
                    break

        finally:
            client.socket.close()
            with self.worker_lock:
                self.current_worker -= 1
                print(f"--- current workers: {self.current_worker}")

    def broadcast_state(self):
        # 确定服务器状态
        if self.current_worker < self.max_workers:
            state_message = "Server State: OPEN"
        else:
            state_message = "Server State: CLOSED"

        # 广播状态给所有在线客户端
        print(f"-> broadcast: \"{state_message}\"")
        disconnected_clients = []
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.socket.sendall(state_message.encode('utf-8'))
                except Exception:
                    # 如果客户端断开连接，记录并移除
                    disconnected_clients.append(client)

            # 移除断开的客户端
            self.clients = [c for c in self.clients if c not in disconnected_clients]

    def background_broadcast(self):
        while self.running:
            time.sleep(self.broadcast_interval) 
            self.broadcast_state()

# 运行服务器
if __name__ == "__main__":
    server = Server("127.0.0.1", 65432, 5, 5)
    server.start()