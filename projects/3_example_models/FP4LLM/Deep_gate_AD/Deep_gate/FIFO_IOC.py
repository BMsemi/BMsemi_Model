class FIFO:
    def __init__(self, size=1024):
        self.size = size
        self.buffer = []

    def push(self, data):
        if len(self.buffer) < self.size:
            self.buffer.append(data)
            
        else:
            raise OverflowError("FIFO overflow")

    def pop(self):
        if self.buffer:
            return self.buffer.pop(0)
        else:
            raise IndexError("FIFO underflow")

    def is_empty(self):
        return len(self.buffer) == 0

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)