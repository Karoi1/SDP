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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
            :Int: the id of model if successfully created
            :None: if fail to create
            :Str: error message
        """
        
        if len(shape) != 4:
            return None, f" !! Model Manager | addModel(): expect len(shape)==4, get {len(shape)}"
        I,W,O,L = shape
        if I is None or W is None or O is None or L is None:
            return None, f" !! Model Manager | addModel(): One of the item in shape is None"
        self.idcounter += 1
        modelID = self.idcounter
        Newmodel = modelItem(shape, id=modelID, split_layer=split_layer)
        self.Listmodel = np.append(self.Listmodel, Newmodel)
        return modelID, ""

    def check_model_exist(self, split_layer):
        """
        check whether there is model of the corresponding split layer
        Parameters:
            :split_layer: the split layer of the model
        Return:
            :Bool: if model found
        """
        item = next((item for item in self.Listmodel if item.split_layer == split_layer), None)
        if item is None:
            return False
        return True
    def getRandomModelID(self, split_layer):
        """
        Get the id of a random model which is of the corresponding split layer
        Parameters:
            :split_layer: the split layer of model
        Return:
            :Int: Model ID, if the model is found
            :None: if no model found
        """
        L = np.array([item.id for item in self.Listmodel if item.split_layer==split_layer])
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
        if full.split_layer < 1:
            return None, f" !! Model Manager | getClientModel: model.split_layer = {full.split_layer} < 1"
        
        I,W,_,_ = full.shape
        split_layer = full.split_layer
        subShape = np.array([I,W,W,split_layer])
        CM = modelItem(subShape)
        #print(CM.model)
        # copy the full model weight into sub model
        for i in range(split_layer):
            #print(CM.model.layers[i].weight.shape)
            CM.model.layers[i].weight.data.copy_(full.model.layers[i].weight.data)
            CM.model.layers[i].bias.data.copy_(full.model.layers[i].bias.data)
        return CM, ""
    
    def forward_for_id(self, id, smashed_data, labels):
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
        model.eval()
        split_layer = item.split_layer
        with torch.no_grad():
            outputs = model.forward(smashed_data, split_layer)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs,labels)
            item.test_loss_hist = np.append(item.test_loss_hist, loss.item())

            _, predicted = torch.max(outputs,1)
            correct = (predicted == labels).sum().item()
            accuracy = correct/labels.size(0)
            item.train_accuracy_hist = np.append(item.train_accuracy_hist, accuracy)
        return outputs, ""
    
    def train_for_id(self, id, smashed_data, labels):
        """
        Train the model of corresponding id with smashed data(at split layer). 
        Parameters:
            :id: the id of the model
            :smashed_data: smashed data at split layer
            :labels: labels of the smashed data
        Return:
            :Tensor: the gradient at split layer if model found
            :None: if no model found
            :Str: error message
        """
        item, error = self.getModel(id)
        if item is None:
            return None, error
        model = item.model
        optimizer = item.optimizer
        lock = item.lock
        split_layer = item.split_layer
        with lock:
            smash_data = torch.tensor(smashed_data, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            criterion = nn.CrossEntropyLoss()

            outputs = model.forward(smash_data, split_layer)
            loss = criterion(outputs, labels)
            # record the loss
            item.train_loss_hist = np.append(item.train_loss_hist, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record the accuracy
            _, predicted = torch.max(outputs,1)
            correct = (predicted == labels).sum().item()
            accuracy = correct/labels.size(0)
            item.train_accuracy_hist = np.append(item.train_accuracy_hist, accuracy)

        return self.get_split_layer_gradient(id)

    def get_split_layer_gradient(self, id):
        """
        get the gradient at split layer for the model of corresponding id.
        Parameters:
            :id: the id of the model
        Return:
            :Tensor: the gradient at split layer if model found
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
        self.train_accuracy_hist = np.array([],dtype=np.float32)
        self.train_loss_hist = np.array([],dtype=np.float32)
        self.test_accuracy_hist = np.array([],dtype=np.float32)
        self.test_loss_hist = np.array([],dtype=np.float32)
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
        self.trainSD = None
        self.trainL = None
        self.testSD = None
        self.testL = None
        self.gradient = None
        self.batchN = None
        self.subscribe = subscribe
        self.modelID = None
    def is_no_info(self):
        attrList = ['trainSD', 'trainL', 'testSD', 'testL', 'gradient']
        for i in attrList:
            if getattr(self, i) is not None:
                return False
        return True
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
        self.shape = [input_size,modelW,output_size,modelL]
        self.MM = modelManager()
    def start(self):
        print("creating socket...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_connections)
        self.server_socket.settimeout(1)

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

        model_updater_thread = threading.Thread(target=self.model_updator)
        model_updater_thread.daemon = True
        model_updater_thread.start()

        gradient_distributor_thread = threading.Thread(target=self.gradient_distributor)
        gradient_distributor_thread.daemon = True
        gradient_distributor_thread.start()

        print("========********[ Server On Board ]********========")
        print(f" === Listen at {self.host}: {self.port}")

        try:
            while self.running:
                try:
                    client_socket, client_addr = self.server_socket.accept()
                    print(f"-> Listen: connection from: {client_addr}")
                except socket.timeout:
                    continue

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
    

    def generate_messages(self, type, message=None):
        if type == "loginState":
            return json.dumps({"type": type, "value": message})
        if type == "model":
            binary_data = self.tensor_to_byte(message[0].state_dict())
            print(len(binary_data))
            shape = message[1:]
            #print(shape)
            return json.dumps({"type": type, "shape": shape, "binary_data": binary_data})
        if type == "hb":
            return json.dumps({"type": type}).encode('utf-8')
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
                        if client.is_no_info():
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
        
    def model_tester(self):
        while self.running:
            time.sleep(0.5)
            with self.clients_lock:
                #print("update model check")
                for client in self.clients:
                    #print(f"ip: {client.ip_address}")
                    if client.testSD is not None and client.testL is not None:
                        id = client.modelID
                        smashed_data = client.testSD
                        labels = client.testL
                        self.MM.forward_for_id(id,smashed_data,labels)
                        client.testSD = None
                        client.testL = None
                        #print(client.ip_address,"train")
    def model_distributor(self):
        while self.running:
            time.sleep(self.DMCD)
            with self.clients_lock:
                counter = 0
                for client in self.clients:
                    if client.state == "online waiting":
                        #print(client.ip_address)
                        id = client.modelID
                        item, e = self.MM.getClientModel(id)
                        if item is None:
                            print(e)
                        I,W,O,L = item.shape
                        sub_model = item.model
                        m = self.generate_messages("model", [sub_model,int(I),int(W),int(L)])
                        self.send_client_mes(client, m)
                        client.state = "working"
                        counter += 1
            if counter != 0:
                print(f"-> Model distributor: send model to {counter} clients")

    
    def tensor_to_byte(self, tensor):
        binary_data = io.BytesIO()
        torch.save(tensor, binary_data)
        return base64.b64encode(binary_data.getvalue()).decode("utf-8")
    def model_updator(self):
        while self.running:
            time.sleep(0.1)
            with self.clients_lock:
                for client in self.clients:
                    #print(f"ip: {client.ip_address}")
                    if client.trainSD is not None and client.trainL is not None:
                        id = client.modelID
                        smashed_data = client.trainSD
                        labels = client.trainL
                        gradient, _ = self.MM.train_for_id(id,smashed_data,labels)
                        client.gradient = gradient
                        client.trainSD = None
                        client.trainL = None
                        #print(client.ip_address,"train")

    def gradient_distributor(self):
        while self.running:
            time.sleep(0.1)
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
            # client want to compute 0~n-1 layer, then search for split layer n
            prefer = data.get('prefer')
            client.prefer = prefer
            #print("prefer",prefer)
            if self.MM.check_model_exist(prefer):
                client.modelID = self.MM.getRandomModelID(prefer)
            else:
                id, error = self.MM.addModel(self.shape, prefer)
                if id is None:
                    print(f" !! parse_handle_mes | data: {data} | error: {error}")
                client.modelID = id

            if client.state == "online":
                client.state = "online waiting"
                
            return
        if type == "train SDL":
            if client.state == "working":
                client.trainSD = data.get('SD')
                client.trainL = data.get('L')
                client.batchN = data.get('batchN')
                print(f"received batch {len(client.trainL)}")
            else:
                print(f"~ parse_handle_mes: receive train SDL for non working client | state: {client.state}, ip: {client.ip_address}")
            return
        if type == "test SDL":
            client.testSD = data.get('SD')
            client.testL = data.get('L')
            batchN = data.get('batchN')
            return
        if type == "End":
            print("end")
            item,_ = self.MM.getModel(1)
            print(item.split_layer)
            train_loss = item.train_loss_hist
            train_accuracy = item.train_accuracy_hist
            test_loss = item.test_loss_hist
            test_accuracy = item.test_accuracy_hist
            print(len(train_loss))
            print(len(train_accuracy))
            print(len(test_loss))
            print(len(test_accuracy))
            # 创建一个 2x2 的 subplot 布局
            plt.figure(figsize=(14, 10))

            # 第一个 subplot：训练损失
            plt.subplot(2, 2, 1)  # 2行2列的第1个位置
            plt.plot(train_loss, label='Train Loss', marker='o')
            plt.title('Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # 第二个 subplot：训练准确率
            plt.subplot(2, 2, 2)  # 2行2列的第2个位置
            plt.plot(train_accuracy, label='Train Accuracy', marker='o', color='orange')
            plt.title('Train Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

            # 第三个 subplot：测试损失
            plt.subplot(2, 2, 3)  # 2行2列的第3个位置
            plt.plot(test_loss, label='Test Loss', marker='o', color='green')
            plt.title('Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # 第四个 subplot：测试准确率
            plt.subplot(2, 2, 4)  # 2行2列的第4个位置
            plt.plot(test_accuracy, label='Test Accuracy', marker='o', color='red')
            plt.title('Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

            # 调整子图间距
            plt.tight_layout()

            # 显示整个图表
            plt.savefig('img.png')
            plt.close()
        print(f"~ parser: unknown data | ip: {client.ip_address} | d: {data}")

    def listen_client_mes(self, client):
        try:
            with self.worker_lock:
                self.current_worker += 1
                print(f"+++ current workers: {self.current_worker}")

            while self.running:
                try:
                    mes = client.socket.recv(pow(2,24)).decode('utf-8')
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