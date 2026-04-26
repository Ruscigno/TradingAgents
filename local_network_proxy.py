import socket
import threading
import sys

def handle_client(client_socket, target_host, target_port):
    try:
        remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        remote_socket.connect((target_host, target_port))
        
        def forward(src, dst):
            try:
                while True:
                    data = src.recv(4096)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception:
                pass
            finally:
                src.close()
                dst.close()

        threading.Thread(target=forward, args=(client_socket, remote_socket)).start()
        threading.Thread(target=forward, args=(remote_socket, client_socket)).start()
    except Exception as e:
        print(f"Failed to connect to remote server: {e}")
        client_socket.close()

def main():
    if len(sys.argv) != 3:
        target_host = "192.168.0.156"
        target_port = 11434
    else:
        target_host = sys.argv[1]
        target_port = int(sys.argv[2])

    listen_host = "127.0.0.1"
    listen_port = 11434

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((listen_host, listen_port))
        server_socket.listen(5)
        print(f"Proxy listening on {listen_host}:{listen_port}")
        print(f"Forwarding to {target_host}:{target_port}")
        
        while True:
            client, addr = server_socket.accept()
            threading.Thread(target=handle_client, args=(client, target_host, target_port)).start()
            
    except KeyboardInterrupt:
        print("\nShutting down proxy.")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()
