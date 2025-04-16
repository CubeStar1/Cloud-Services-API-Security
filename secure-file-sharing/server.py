import socket
import json
import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='syslog.log',
    level=logging.INFO,
    format='%(message)s',
    filemode='a'  # Append to the file instead of overwriting
)


def syslog_server():
    """UDP server to receive and log syslog messages."""
    try:
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)  # 256KB buffer

        # Bind to localhost:1514
        server_address = ('0.0.0.0', 1514)
        sock.bind(server_address)
        print(f"Starting syslog server on {server_address}")

        while True:
            try:
                # Receive data
                data, _ = sock.recvfrom(65535)
                message = data.decode('utf-8')

                # Only log JSON messages
                if "SaaS_Monitor - " in message:
                    # Extract the JSON part
                    json_start = message.find("{")
                    if json_start != -1:
                        json_str = message[json_start:]
                        try:
                            # Validate JSON
                            json_obj = json.loads(json_str)
                            # Log the validated JSON
                            logging.info(json_str)
                            print(f"Logged: {json_str}")
                        except json.JSONDecodeError:
                            print(f"Invalid JSON message received: {json_str}")
                    else:
                        # If no JSON found but still has SaaS_Monitor, log as plain text
                        logging.info(message)
                        print(f"Logged plain text: {message}")
            except json.JSONDecodeError:
                print(f"Invalid JSON message received")
            except Exception as e:
                print(f"Error processing message: {e}")

    except Exception as e:
        print(f"Server error: {e}")
    finally:
        sock.close()


if __name__ == '__main__':
    # Write a startup marker to the log file instead of clearing it
    with open('syslog.log', 'a') as f:
        f.write(f'===== Syslog Server Started at {datetime.datetime.now()} =====\n')
    syslog_server()