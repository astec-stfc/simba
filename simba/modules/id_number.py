import time
import zmq
import json


class ZmqClient(object):

    def __init__(self, port=5557, host="localhost"):
        super(ZmqClient, self).__init__()
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
    client = ZmqClient()
    print(client.get_id())
    exit()


from simba._compat import deprecated_aliases  # noqa: E402

__getattr__ = deprecated_aliases(
    __name__,
    globals(),
    {
        "zmqClient": "ZmqClient",
    },
)
