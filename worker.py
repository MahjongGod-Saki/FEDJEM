import pickle
import paramiko
import threading
from paramiko.channel import ChannelFile
from multiprocessing.connection import Listener

master_addr = "172.17.226.11"
master_port = 29500

workers = [
    dict(rank=1, name="worker-1", addr="172.17.226.8"),
    dict(rank=2, name="worker-2", addr="172.17.226.9"),
    dict(rank=3, name="worker-3", addr="172.17.226.10"),
]


def create_script(shared_code: str):
    return f"""
import pickle
from multiprocessing.connection import Listener, Client, Connection
from multiprocessing.pool import ThreadPool

class Worker:
    def __init__(self):
        self._functions = dict()

    def register(self, func):
        self._functions[func.__name__] = func
        return func

    def start(self, address, authkey):
        connection = Client(address, authkey=authkey)
        try:
            while True:
                func_name, args, kwargs = pickle.loads(connection.recv())
                try:
                    r = self._functions[func_name](*args, **kwargs)
                    connection.send(pickle.dumps(r))
                except Exception as e:
                    connection.send(pickle.dumps(e))
                    raise e
        except EOFError:
            pass

worker = Worker()
register = worker.register
{shared_code}

print("worker starting")
worker.start(('{master_addr}', {master_port}), b"authkey")
print("worker exited")
"""


def print_output(name: str, output: ChannelFile):
    lines = output.read().decode("utf-8").splitlines()
    lines = map(lambda x: name + " >>> " + x, lines)
    print("\n".join(lines))


def create_worker(worker: dict, shared_code: str):
    print(
        f"creating worker {worker['name']} (rank={worker['rank']}, addr={worker['addr']})")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=worker["addr"], port=22,
                   username="root", key_filename="/root/.ssh/id_rsa")

    stdin, stdout, stderr = client.exec_command(
        "source /root/miniconda3/bin/activate && python -")

    stdin.write(create_script(shared_code))
    stdin.close()

    print_output(worker["name"] + " stdout", stdout)
    print_output(worker["name"] + " stderr", stderr)

    client.close()


class Master:
    def __init__(self, worker_count):
        self.worker_count = worker_count
        self.connections = []
        self.listener = None
        self.bytes_sent = 0
        self.bytes_recved = 0

    def start(self, address, authkey):
        self.listener = Listener(address, authkey=bytes(authkey))
        while len(self.connections) < self.worker_count:
            client = self.listener.accept()
            self.connections.append(client)

    def close(self):
        self.listener.close()

    def rpc(self, worker, name, *args, **kwargs):
        conn = self.connections[worker]
        send = pickle.dumps((name, args, kwargs))
        self.bytes_sent += len(send)
        conn.send(send)
        recv = conn.recv()
        self.bytes_recved += len(recv)
        result = pickle.loads(recv)
        if isinstance(result, Exception):
            raise result
        return result


def init_rpc(shared_code: str):

    for worker in workers:
        thread = threading.Thread(
            target=create_worker, args=(worker, shared_code))
        thread.start()

    print("waiting workers")

    master = Master(len(workers))
    master.start((master_addr, master_port), b"authkey")

    print("all workers ready")

    return master


if __name__ == "__main__":

    shared_code = """
@register
def add(a, b):
    return a + b
"""

    master = init_rpc(shared_code)

    print(master.rpc(0, "add", 1, 2))
    print(master.rpc(0, "add", 1, 2))
    print(master.rpc(0, "add", 1, 2))
