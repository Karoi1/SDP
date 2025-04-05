import socket
import datetime

def main():
    HOST = '127.0.0.1'
    PORT = 65432
    
    for i in range(3):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((HOST, PORT))
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                mes = current_time.encode()
                s.sendall(mes)
                print(f"send time: {current_time}")
                input()
            except socket.error as e:
                print(f"connection failure: {e}")

if __name__ == "__main__":
    main()