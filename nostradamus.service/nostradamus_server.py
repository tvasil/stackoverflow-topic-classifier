import logging
from concurrent import futures

import grpc
import nostradamus_pb2
import nostradamus_pb2_grpc
from grpc_reflection.v1alpha import reflection
from so_tag_classifier_prediction import binarize_ys, predict, text_prepare, tokenize_and_stem

_MODEL_FNAME = "../models/2020_11_08_rs_model_and_mlb.pkl"


def concatenate_title_body(title: str, body: str) -> str:
    """
    Predictions are made on  the concatenated title plus body of the post
    """
    return title + " " + body


class TopicLabelerServicer(nostradamus_pb2_grpc.TopicLabelerServicer):
    """
    Provides methods that implement the functionality of predicting SO topic labels
    """

    def Predict(self, request, context):
        prediction = predict(sentence=concatenate_title_body(request.title, request.body), fname=_MODEL_FNAME)
        return nostradamus_pb2.TopicLabelReply(prediction=prediction)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nostradamus_pb2_grpc.add_TopicLabelerServicer_to_server(TopicLabelerServicer(), server)
    SERVICE_NAMES = (
        nostradamus_pb2.DESCRIPTOR.services_by_name["TopicLabeler"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
