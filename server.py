from scipy.special import lambertw
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import socket
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

n = 3   # number of users
epochs = 10

B = 100         # Bandwidth
T = np.random.rand(n)           # total time limit
L = 100         # data size
N0 = 1e-9       # noise
nu = 1    # Lagrange multiplier
beta = np.random.randint(2, size=n) # selected client
gamma = np.zeros(n)                 # bandwidth partition for clients
h = np.random.rand(n)               # channel gain
t = np.zeros(n)                    # transimition time
max_search = 2**(n//2)                     # max iter for finding beta
search_tol = 0.01                   # tolerance for sum(gamma)=1
max_iter = 20                       # max iter of finding (beta,gamma,t)
tol = 0.01

gamma_hist = []


class clientDevice:
    def __init__(self, conn, addr, id):
        self.conn = conn
        self.addr = addr
        self.id = id
        self.beta = 0
        self.T = -1
        self.h = -1
    def set_time_transfer(self,T):
        self.T = T
    def set_channel_gain(self, h):
        self.h = h
    def set_beta(self, beta):
        self.beta = beta
    def __str__(self):
        return f"Device {self.id}, addr: {self.addr}, beta: {self.beta}, T: {self.T}, h: {self.h}"
    


def process(client):
    print("Connected by", client.addr)
    try:
        data = client.conn.recv(1024).decode()
        print("Received:", data)
        client_time = datetime.strptime(data, '%Y-%m-%d %H:%M:%S.%f')
        server_time = datetime.now()

        time_transfer = (server_time - client_time).total_seconds()
        client.set_time_transfer(time_transfer)
        print("transfer time: ", time_transfer)
    except Exception as e:
        print(f"error processing client: {e}")
    finally:
        client.conn.close()
        print("Client disconnected")


        



def main():
    global gamma, beta, B, T, L, N0, h, nu, max_search, search_tol, max_iter, tol

    max_n = 3
    id_counter = 0
    HOST = '127.0.0.1'
    PORT = 65432
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))

    server_socket.listen(max_n)
    print("Server is running...waiting for client connect")

    with ThreadPoolExecutor(max_workers=max_n) as pool:
        futurelist = []
        counter = 0
        clientlist = []
        while True:
            conn, addr = server_socket.accept()
            id_counter += 1
            client = clientDevice(conn, addr, id_counter)
            client.set_channel_gain(0.1*np.random.randint(1,11))
            clientlist.append(client)
            future = pool.submit(process, client)
            futurelist.append(future)
            if id_counter == max_n:
                break
    
    for i in range(max_n):
        h[i] = clientlist[i].h
        T[i] = clientlist[i].T

    for i in range(epochs):
        gamma,beta,t,nu = EERRA_iter(gamma, beta, B, T, L, N0, h, nu, max_search, search_tol, max_iter, tol)
    model_indices = list(range(1, epochs+1))
    plt.figure(figsize=(10, 5))
    for i in range(n):
        gamma_i = [gamma_k[i] for gamma_k in gamma_hist]
        plt.plot(model_indices, gamma_i, label=f'Client {i+1}')
    plt.legend()
    plt.xlabel('Training Round')
    plt.ylabel(r'$\gamma$ (Bandwidth Allocation)')
    plt.title('Bandwidth Allocation Over Training Rounds')
    plt.xticks(model_indices)

    # 调整子图间距
    plt.tight_layout()

    # 显示图形
    plt.show()
    



def compute_gamma_tk(beta, B, T, L, N0, h, nu):
    gamma = np.zeros(n)
    t = np.zeros(n)
    

    for k in range(n):
        if beta[k] == 1:
            term_inside_w = (h[k]**2 * nu - B * T[k] * N0) / (B * T[k] * N0 * np.e)
            w_value = lambertw(term_inside_w).real 
            gamma[k] = (beta[k] * L * np.log(2)) / (B * T[k] * (1 + w_value))
            t[k] = T[k]
        if beta[k] == 0:
            gamma[k] = 0
            t[k] = T[k]
    return gamma, t

def update_beta(gamma, B, T, L, N0, h, nu):
    beta = np.zeros(n)
    for k in range(n):
        term_inside_log = (nu* h[k]**2) / (N0 * L * np.log(2))
        log_term = np.log(term_inside_log)
        beta_k = min(max((gamma[k] * B * T[k] * log_term) / L, 0), 1)
        if beta_k >= 0.5:
            beta[k] = 1
        if beta_k < 0.5:
            beta[k] = 0
    return beta

def gamma_sum(nu, beta, B, T, L, N0, h):
    sum_gamma = 0
    for k in range(n):
        if beta[k] == 1:
            term_inside_w = (h[k]**2 * nu - B * T[k] * N0) / (B * T[k] * N0 * np.e)
            w_value = lambertw(term_inside_w).real
            gamma_k = (beta[k] * L * np.log(2)) / (B * T[k] * (1 + w_value))
            sum_gamma += gamma_k
    return sum_gamma-1

def EERRA_iter(gamma, beta, B, T, L, N0, h, nu, max_search, search_tol, max_iter, tol):
    global gamma_hist
    #T = np.random.rand(n)
    for i in range(max_iter):
        previous = gamma, beta, B, T, L, N0, h, nu
        for j in range(max_search):
            if abs(gamma_sum(nu, beta, B, T, L, N0, h)) <= search_tol:
                #print(f"{j} time used, gamma sum: {gamma_sum(nu, beta, B, T, L, N0, h)}")
                break
            #print(f"searching beta...gamma sum: {gamma_sum(nu, beta, B, T, L, N0, h)}")
            beta = np.random.randint(2,size=n)
            if np.sum(beta) == 0:
                beta[np.random.randint(n)] = 1
            nu, = fsolve(gamma_sum, x0=1.0, args=(beta, B, T, L, N0, h))

        gamma, t = compute_gamma_tk(beta, B, T, L, N0, h, nu)
        beta = update_beta(gamma, B, T, L, N0, h, nu)

        #check convergence
        if np.sum(np.abs(previous[0] - gamma)) <= tol:
            #print(f"{i} time to converge beta")
            break
    print(f"beta: {beta}")
    print(f"gamma: {gamma}")
    if np.sum(beta) == 0:
        return EERRA_iter(gamma, beta, B, T, L, N0, h, nu, max_search, search_tol, max_iter, tol)
    gamma_hist.append(gamma)
    return gamma, beta, t, nu

if __name__ == "__main__":
    main()