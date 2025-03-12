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
    def __init__(self, allocate_metric="one to one"):
        self.allocate_metric=allocate_metric
        self.idcounter = 0
        self.ListmodelLock = threading.Lock()
        self.Listmodel = []
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
        with self.ListmodelLock:
            self.Listmodel.append(Newmodel)
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
        item = next((item for item in self.Listmodel if item.id == id), None)
        if item is None:
            return None, f" !! Model Manager | getModel(): no model of {id} found"
        if item.reference != -1:
            return self.getModel(item.reference)
        return item, ""
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
        with item.lock:
            item.testN += len(labels)
        with torch.no_grad():
            smashed_data = torch.tensor(smashed_data, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            outputs = model.forward(smashed_data, split_layer)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs,labels)
            item.test_loss_hist = np.append(item.test_loss_hist, loss.item())

            _, predicted = torch.max(outputs,1)
            correct = (predicted == labels).sum().item()
            accuracy = correct/labels.size(0)
            item.test_accuracy_hist = np.append(item.test_accuracy_hist, accuracy)
        #print("here")
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
            item.trainN += len(labels)
            smash_data = torch.tensor(smashed_data, dtype=torch.float32, requires_grad=True)
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
        #print(smash_data.grad.shape)
        return smash_data.grad, ""

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
        split_layer = item.split_layer
        gradients = []
        i = 0
        #print("get split = ",split_layer)
        for name, param in model.named_parameters():
            #print(param.shape)
            if "weight" in name:
                if i == split_layer:
                    gradients.append(param.grad.data.clone())
                    #print(gradients[0].shape)
                    break
                i += 1
        return gradients[0], ""

    def integrate_model_id(self, id_list, split_layer=1, metric='avg', delete=False):

        """
        Integrate the model in id_list
        !!! old models will be deleted if set delete=True, default delete=False
        The model must be of the same shape
        Parameters:
            :id_list: list of id of models to be integrated
            :split_layer: the split layer of the new model
            :metric: the metric used to compute the new model parameters
            :Note: metric type:= 'avg' | 'Navg' | 'Wloss' | 'Wacc'
        Returns:
            :modelItem(): if success, the new model
            :None: if fail
            :Str: error message
        """
        item_list = []
        #append the model item into list
        for id in id_list:
            item, error = self.getModel(id)
            if item is None:
                return None, "MM: integrate_model: "+error
            item_list.append(item)
        
        # prepare all locks
        lock_list = [item.lock for item in item_list]
        model_list = [item.model for item in item_list]
        # check if all model shape is same
        shape_list = [item.shape for item in item_list]
        for i in range(len(shape_list)):
            if shape_list[i] != shape_list[0]:
                return None, f"MM: integrate_model: model shape is not same, [0] = {shape_list[0]}, [{i}] = {shape_list[i]}"
        
        # get all locks
        for l in lock_list:
            l.acquire()

        newModelItem = None
        if metric == 'avg':
            newModelID,_ = self.addModel(shape_list[0],split_layer=split_layer)
            newModelItem,_ = self.getModel(newModelID)
            newModel = newModelItem.model
            for key in newModel.state_dict().keys():
                avg_param = 0
                for model in model_list:
                    avg_param += model.state_dict()[key]
                avg_param /= len(model_list)
                newModel.state_dict()[key].copy_(avg_param)
            for item in item_list:
                item.reference = newModelID

        for l in lock_list:
                l.release()
        if newModelItem is not None and delete: #note, cannot do now because you havn't direct client to the new model by the code following
            with self.ListmodelLock:
                pass
                #self.Listmodel = [item for item in self.Listmodel if item.id not in id_list]
        if newModelItem is None:
            return None, f"MM: integrate_model: Unknown metric: {metric}"
        

        return newModelItem, ""

    def copy_model(self, id, split_layer):
        """
        Copy the model of the id to a new model, specify the split layer of the new model
        Parameters:
            :id: the old model id
            :split_layer: specify the split layer of the new model
        Returns:
            :Int: the new model id if success
            :None: if fail
            :Str: error message
        """
        item,error = self.getModel(id)
        if item is None:
            return None, error
        
        # create a new model
        shape = item.shape
        _,_,_,L = shape
        newModelID = self.addModel(shape, split_layer)
        newModelItem = self.getModel(newModelID)
        newModel = newModelItem.model
        for i in range(L):
            newModel.layers[i].weight.data.copy_(item.model.layers[i].weight.data)
            newModel.layers[i].bias.data.copy_(item.model.layers[i].bias.data)
        return newModelID, ""

    def allocate_model_to_client(self, shape, client):
        """
        Distribute the model to client. store the model id into client.modelID
        Parameters:
            :client: the client object
        Returns Nothing
        """
        prefer = client.prefer
        # single: only one model for each different split layer
        if self.allocate_metric == "single":
            if self.check_model_exist(prefer):
                client.modelID,_ = self.getRandomModelID(prefer)
            else:
                client.modelID,_ = self.addModel(shape, prefer)
            return
        # each client is allocated with one model
        if self.allocate_metric == "one to one":
            print("one to one")
            client.modelID,_ = self.addModel(shape, prefer)
            return

class modelItem:
    def __init__(self, shape, id=-1, batch_size=128, lr=0.0001, split_layer=0):
        self.id = id
        self.split_layer = split_layer
        self.shape = shape
        self.model = None
        self.lr = lr
        self.batch_size = batch_size
        self.trainN = 0
        self.testN = 0
        self.reference = -1
        self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.lock = threading.Lock()
        self.train_accuracy_hist = np.array([],dtype=np.float32)
        self.train_loss_hist = np.array([],dtype=np.float32)
        self.test_accuracy_hist = np.array([],dtype=np.float32)
        self.test_loss_hist = np.array([],dtype=np.float32)
    def build_model(self):
        I,W,O,L = self.shape
        if self.split_layer >= L:
            self.split_layer = L-1
        self.model = DynamicMLP(I,W,O,L)
        

    

class DynamicMLP(nn.Module):
    def __init__(self, I, W, O, L):
        super(DynamicMLP, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(I, W)])
        for _ in range(L - 2): 
            self.layers.append(nn.Linear(W, W))
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
        self.InfoLock = threading.Lock()
        self.TrainLock = threading.Lock()
        self.TestLock = threading.Lock()
        self.GLock = threading.Lock()
        self.trainSD = []
        self.trainL = []
        self.testSD = []
        self.testL = []
        self.gradient = torch.tensor([])
        self.batchN = 0
        self.subscribe = subscribe
        self.modelID = None
    def is_no_info(self):
        # TODO not none but empty list
        attrList = ['trainSD', 'trainL', 'testSD', 'testL', 'gradient']
        for i in attrList:
            if getattr(self, i) is not None:
                return False
        return True
    
class Server:                    
    def __init__(self, host, port, max_connections, max_workers, max_wait_queue=5, broadcast_interval=60, input_size=784, output_size=10, modelL=4, modelW=128):
        
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
        self.mhandlerpool = concurrent.futures.ThreadPoolExecutor(max_workers=32)
        
        # list of all connected clients
        self.clients = []
        self.running = True

        # model_distributor CD
        self.DMCD = 5
        self.UDCD = 5
        # specify the limited time of one epoch (second) to run fedavg
        self.epoch_time = 1

        self.inputSize = input_size
        self.outputSize = output_size
        self.modelL = modelL
        self.modelW = modelW
        self.shape = [input_size,modelW,output_size,modelL]
        self.MM = modelManager()


    def start(self):
        """To Start all of the things"""
        print("creating socket...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_connections)
        self.server_socket.settimeout(1)

        #self.start_threads()  TODO
        print("initialize brocast thread...")
        # 启动定时广播线程
        broadcast_thread = threading.Thread(target=self.background_broadcast)
        broadcast_thread.daemon = True
        #broadcast_thread.start()

        #print("initialize checker thread...")
        # check queue thread
        check_queue_thread = threading.Thread(target=self.queue_checker)
        check_queue_thread.daemon = True
        check_queue_thread.start()

        # check client thread
        check_client_thread = threading.Thread(target=self.cleaner)
        check_client_thread.daemon = True 
        check_client_thread.start()

        model_distributor_thread = threading.Thread(target=self.model_distributor)
        model_distributor_thread.daemon = True
        model_distributor_thread.start()

        model_updater_thread = threading.Thread(target=self.model_updator)
        model_updater_thread.daemon = True
        model_updater_thread.start()

        model_tester_thread = threading.Thread(target=self.model_tester)
        model_tester_thread.daemon = True
        model_tester_thread.start()

        gradient_distributor_thread = threading.Thread(target=self.gradient_distributor)
        gradient_distributor_thread.daemon = True
        gradient_distributor_thread.start()
        

        print("========********[ Server On Board ]********========")
        print(f" === Listen at {self.host}: {self.port}")

        try:
            # while running
            while self.running:
                try:
                    # receive socket
                    client_socket, client_addr = self.server_socket.accept()
                    print(f"-> Listen: connection from: {client_addr}")
                except socket.timeout:
                    continue
                
                # create client object
                client = Client(client_addr, socket=client_socket)
                with self.clients_lock:
                    # with lock, append object to list
                    self.clients.append(client)
                
                with self.worker_lock:
                    #check if there is space available
                    if self.current_worker < self.max_workers:
                        # There is enough space, forward to thread to listen
                        m = self.generate_messages("loginState", "OK")
                        self.send_client_mes(client, m)
                        self.receiverpool.submit(self.listen_client_mes, client)
                        client.state = "online"
                        print(f"-> Listen: {client_addr} direct to listen_client_mes()")
                    else:

                        if self.wait_queue.full():
                            # if queue if full, disconnect
                            m = self.generate_messages("loginState", "FULL")
                            self.send_client_mes(client, m)
                            client.socket.close()
                            print(f"-> Listen: Queue Full, connection break: {client_addr}")
                        else:
                            # if queue has space, put in queue
                            client.state = "queueing"
                            m = self.generate_messages("loginState", "Queueing")
                            self.send_client_mes(client, m)
                            self.wait_queue.put(client)
                            print(f"-> Listen: {client_addr} direct to queue")


        except KeyboardInterrupt:
            print("Server stopped.")
        finally:
            # shut down all
            self.running = False
            self.server_socket.close()
            self.receiverpool.shutdown()
            self.senderpool.shutdown()
            self.mhandlerpool.shutdown()
            broadcast_thread.join()
            check_queue_thread.join()
            model_tester_thread.join()
            check_client_thread.join()
            model_distributor_thread.join()
            model_updater_thread.join()
            model_tester_thread.join()
            gradient_distributor_thread.join()
            
            
    def start_threads(self):
        # TODO
        print("initialize brocast thread...")
        # 启动定时广播线程
        broadcast_thread = threading.Thread(target=self.background_broadcast)
        broadcast_thread.daemon = True
        #broadcast_thread.start()

        #print("initialize checker thread...")
        # check queue thread
        check_queue_thread = threading.Thread(target=self.queue_checker)
        check_queue_thread.daemon = True
        check_queue_thread.start()

        # check client thread
        check_client_thread = threading.Thread(target=self.cleaner)
        check_client_thread.daemon = True 
        check_client_thread.start()

        model_distributor_thread = threading.Thread(target=self.model_distributor)
        model_distributor_thread.daemon = True
        model_distributor_thread.start()

        model_updater_thread = threading.Thread(target=self.model_updator)
        model_updater_thread.daemon = True
        model_updater_thread.start()

        model_tester_thread = threading.Thread(target=self.model_tester)
        model_tester_thread.daemon = True
        model_tester_thread.start()

        gradient_distributor_thread = threading.Thread(target=self.gradient_distributor)
        gradient_distributor_thread.daemon = True
        gradient_distributor_thread.start()

        while True:
            if not self.running:
                broadcast_thread.join()
                check_queue_thread.join()
                model_tester_thread.join()
                check_client_thread.join()
                model_distributor_thread.join()
                model_updater_thread.join()
                model_tester_thread.join()
                gradient_distributor_thread.join()


    def generate_messages(self, type, message=""):
        """
        Generate message for the corresponding type
        Parameters:
            :type: (Str) the message type
            :message: The message
        Return:
            :json: processed message
        """
        if type == "loginState":
            return json.dumps({"type": type, "value": message})
        if type == "model":
            binary_data = self.tensor_to_byte(message[0].state_dict())
            print(f"Size of model: {len(binary_data)} bytes")
            lr = message[1]
            shape = message[2:]
            #print(shape)
            return json.dumps({"type": type, "shape": shape, "lr": lr, "binary_data": binary_data})
        if type == "hb":
            return json.dumps({"type": type}).encode('utf-8')
        if type == "gradient":
            binary_data = self.tensor_to_byte(message)
            return json.dumps({"type": type, "G": binary_data})
        
        print(" !! generate_messages | invalid type: \"{type}\", m: \"{message}\"")

    def cleaner(self):
        """Thread: check and swap the disconnected clients"""
        while self.running:
            # 10s 1 round
            time.sleep(10)

            offline_clients = []
            m = self.generate_messages("hb")
            with self.clients_lock:
                # get the lock 
                for client in self.clients:
                    try:
                        # send a heart beat
                        client.socket.sendall(m)
                    except (socket.error, OSError):
                        # if disconnected, append it to list
                        if client.is_no_info():
                            offline_clients.append(client)

                # delete the clients from list
                self.clients = [c for c in self.clients if c not in offline_clients]

            # close sockets of off line client
            for client in offline_clients:
                try:
                    client.socket.close()
                except Exception:
                    pass
            if len(offline_clients) != 0:
                print(f"-> Cleaner: Removed {len(offline_clients)} offline client")


    def queue_checker(self):
        """Thread: Check the wait queue"""
        while self.running:
            time.sleep(1)
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
                            print(f" !! queue_checker() Error: {e}")
                            
    
    def send_client_mes(self, client:Client, mes):
        """
        Send message to client
        Parameters:
            :client: client
            :mes: message
        Return Nothing
        """
        try:
            client.socket.sendall(mes.encode('utf-8'))
        except Exception as e:
            print(f" !! send_client_mes Error | ip: {client.ip_address} | e: {e} | m: {1}")
        
    def model_tester(self):
        """Thread for testing model"""
        while self.running:
            time.sleep(0.02)
            #print("update model check")
            for client in self.clients:
                #print(f"ip: {client.ip_address}")
                if client.testSD and client.testL:
                    with client.TestLock:
                        id = client.modelID
                        smashed_data = client.testSD.copy()
                        labels = client.testL.copy()
                        client.testSD = []
                        client.testL = []
                    self.MM.forward_for_id(id,smashed_data,labels)
                    print("forward ", len(labels))
                        
    def model_distributor(self):
        """Thread for distributing model"""
        while self.running:
            time.sleep(self.DMCD)
            counter = 0
            for client in self.clients:
                if client.state == "online waiting":
                    #print(client.ip_address)
                    id = client.modelID
                    item,_ = self.MM.getModel(id)
                    
                    #if there is model update
                    if item.reference != -1:
                        client.modelID = item.reference
                        id = item.reference
                    
                    item, e = self.MM.getClientModel(id)
                    if item is None:
                        print(e)
                    I,W,O,L = item.shape
                    sub_model = item.model
                    lr = item.lr
                    m = self.generate_messages("model", [sub_model,lr,int(I),int(W),int(L)])
                    self.send_client_mes(client, m)
                    client.state = "working"
                    counter += 1
            if counter != 0:
                print(f"-> Model distributor: send model to {counter} clients")


    def model_updator(self):
        """Thread for updating model and return gradients"""
        while self.running:
            time.sleep(0.02)
            #print("model update")
            for client in self.clients:
                #print(f"ip: {client.ip_address}")
                if client.trainSD and client.trainL:
                    with client.TrainLock:
                        id = client.modelID
                        item,_ = self.MM.getModel(id)

                        #if there is model update
                        if item.reference != -1:
                            client.modelID = item.reference
                            id = item.reference
                        
                        smashed_data = client.trainSD.copy()
                        labels = client.trainL.copy()
                        client.trainSD = []
                        client.trainL = []
                    gradient, _ = self.MM.train_for_id(id,smashed_data,labels)
                    #print("get G ",gradient.shape)
                    with client.GLock:
                        #print("save G")
                        client.gradient = torch.cat((client.gradient, gradient))
                        #print(client.ip_address,"train")

    def gradient_distributor(self):
        """Thread for distributing gradients"""
        while self.running:
            time.sleep(0.02)
            counter = 0
            #print("G distribute")
            for client in self.clients:
                if client.gradient.numel() != 0:
                    with client.GLock:
                        gradient = client.gradient.clone()
                        client.gradient = torch.tensor([])
                    print("send G", gradient.shape)
                    m = self.generate_messages("gradient", gradient)
                    self.send_client_mes(client, m)
                    counter += 1
            if counter!= 0:
                pass
                #print(f"-> Gradient Distributor: sent gradient to {counter} clients")

    def parse_handle_mes(self, client:Client, data):
        """
        Parse and handle the message from client
        Parameters:
            :client: client object
            :data: the json 
        Return Nothing
        """
        #print(f"From {client.ip_address}: {data}")
        data = json.loads(data)
        type = data.get('type')
        if type == "info":
            client.Tx = data.get('Tx')
            # client want to compute 0~n-1 layer, then search for split layer n
            prefer = data.get('prefer')
            client.prefer = prefer
            #print("prefer",prefer)
            self.MM.allocate_model_to_client(self.shape, client)

            if client.state == "online":
                client.state = "online waiting"
                
            return
        if type == "train SDL":
            if client.state == "working":
                with client.TrainLock:
                    #print("add pkg")
                    client.trainSD += data.get('SD')
                    client.trainL += data.get('L')
                    client.batchN += data.get('batchN')
                #print(f"received batch {len(client.trainL)}")
            else:
                print(f"~ parse_handle_mes: receive train SDL for non working client | state: {client.state}, ip: {client.ip_address}")
            return
        if type == "test SDL":
            with client.TestLock:
                client.testSD += data.get('SD')
                client.testL += data.get('L')
                client.batchN += data.get('batchN')
            return
        if type == "End":
            print("end")
            print(client.batchN)
            error = self.save_model_hist_to_img(client.modelID, f"{client.modelID}.png")
            print("error: ",error)
        print(f"~ parser: unknown data | ip: {client.ip_address} | d: {data}")

    def save_model_hist_to_img(self, id, img_name="img.png"):
        """
        Save the acc, loss history of model of the corresponding id into image
        Parameters:
            :id: the model id
            :img_name: the saved image name
        Return
            :Str: if there is error
        """
        item, error = self.MM.getModel(id)
        if item is None:
            return error
        
        train_loss = item.train_loss_hist
        train_accuracy = item.train_accuracy_hist
        test_loss = item.test_loss_hist
        test_accuracy = item.test_accuracy_hist
        print("save for id: ", id)
        print("train round")
        print(len(train_loss))
        print(len(train_accuracy))
        print("test round")
        print(len(test_loss))
        print(len(test_accuracy))

        plt.figure(figsize=(14, 10))
        plt.title(f"The results of model id: {id}")

        plt.subplot(2, 2, 1) 
        plt.plot(train_loss, label='Train Loss', marker='o')
        plt.title('Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)


        plt.subplot(2, 2, 2)
        plt.plot(train_accuracy, label='Train Accuracy', marker='o', color='orange')
        plt.title('Train Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)


        plt.subplot(2, 2, 3)
        plt.plot(test_loss, label='Test Loss', marker='o', color='green')
        plt.title('Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)


        plt.subplot(2, 2, 4)
        plt.plot(test_accuracy, label='Test Accuracy', marker='o', color='red')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(img_name)
        plt.close()
        return ""


    def listen_client_mes(self, client):
        """Thread for listening client message"""
        try:
            with self.worker_lock:
                # get lock and update worker number
                self.current_worker += 1
                print(f"+++ current workers: {self.current_worker}")

            while self.running:
                # while running
                #print("listen here")
                try:
                    # Receive message and decode
                    mes = client.socket.recv(pow(2,24)).decode('utf-8')
                    if not mes:
                        break
                    #self.parse_handle_mes(client, mes)
                    # forward to a new thread to parse and handle the message
                    self.mhandlerpool.submit(self.parse_handle_mes, client, mes)
                    #print("listen now direct")
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

    def tensor_to_byte(self, tensor):
        """Encode tensor to bytes"""
        binary_data = io.BytesIO()
        torch.save(tensor, binary_data)
        return base64.b64encode(binary_data.getvalue()).decode("utf-8")

    def broadcast_state(self):
        # TODO
        if self.current_worker < self.max_workers:
            state_message = "Server State: OPEN"
        else:
            state_message = "Server State: CLOSED"

        print(f"-> broadcast: \"{state_message}\"")
        disconnected_clients = []
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.socket.sendall(state_message.encode('utf-8'))
                except Exception:
                    pass

    def background_broadcast(self):
        """Thread for broadcasting"""
        while self.running:
            time.sleep(self.broadcast_interval) 
            self.broadcast_state()

    def fedavg(self):
        """Thread for fedavg"""
        while self.running:
            time.sleep(self.epoch_time)
            id_list = []
            for item in self.MM.Listmodel:
                if item.reference == -1:
                    # note, no shape check here. But the shape should be the same for all model now
                    # may check by new attribute "task name"
                    id_list.append(item.id)
            self.MM.integrate_model_id(id_list)
if __name__ == "__main__":  
    #Run
    server = Server("127.0.0.1", 65432, 5, 5)
    server.start()