import time
import zmq
import json


class zmqClient(object):

    def __init__(self, port=5557, host="localhost"):
        super(zmqClient, self).__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.linger = 0
        self.socket.connect("tcp://%s:%s" % (host, port))

    def request(self, msg="get_number"):
        self.socket.send_pyobj(msg)
        message = self.socket.recv_pyobj()
        # print "Received reply ", kwargs, "[", message, "]"
        return message

    def get_id(self):
        return self.request(msg="get_number")

    def reset_id(self):
        return self.request(msg="reset_number")


if __name__ == "__main__":
    client = zmqClient()
    print(client.get_id())
    exit()
