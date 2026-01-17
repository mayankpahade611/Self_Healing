from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["endpoint", "method", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latency of HTTP requests",
    ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made"
)

DRIFT_COUNT = Counter(
    "drift_detected_total",
    "Number of drift detections"
)
