from __future__ import print_function

import argparse

import grpc
import nostradamus_pb2
import nostradamus_pb2_grpc


def run(host, port):
    channel = grpc.insecure_channel("%s:%d" % (host, port))
    stub = nostradamus_pb2_grpc.TopicLabelerStub(channel)
    request = nostradamus_pb2.TopicLabelRequest(
        title="Drawing a diagonal line of my matplotlib scatterplot?",
        body="""I am trying to draw a diagonal line on my figure to demonstrate how my data
            compares to someone else's, so I want a line representing 1:1 relationship.
            I'm trying to use plt.plot to do the line between two points but there is no line on my plot.
            This is my code + the figure. Can anyone tell me why it is not working?""",
    )
    response = stub.Predict(request)
    print("Predicted labels: " + str(response.prediction.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="host name", default="localhost", type=str)
    parser.add_argument("--port", help="port number", default=50051, type=int)
    args = parser.parse_args()
    run(args.host, args.port)
