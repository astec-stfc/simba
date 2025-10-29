import os, sys
import time
import zmq
import threading
import signal


class zmqServer(threading.Thread):

    _id = -1

    def __init__(self, port=5557):
        super(zmqServer, self).__init__()
        self.event = threading.Event()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.linger = 0
        self.port = port

    def run(self):
        self.socket.bind("tcp://*:%s" % (self.port))
        self.socket.linger = 0
        while not self.event.is_set():
            msg = self.socket.recv_pyobj()
            # print 'msg = ', msg
            if msg == "get_number":
                self._id += 1
                self.socket.send_pyobj(self._id)
            elif msg == "reset_number":
                self._id = -1
                self.socket.send_pyobj(self._id)

    def stop(self):
        print("stopping server...")
        self.event.set()
        self.socket.close()
        self.context.destroy()


if __name__ == "__main__":
    server = zmqServer()
    server.daemon = True
    server.start()
    while True:
        time.sleep(1)
