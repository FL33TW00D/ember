from __future__ import annotations
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uvicorn.logging import DefaultFormatter
import uvicorn

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer
import multiprocessing as mp
import time
from datetime import datetime, timedelta
import queue
import threading
import logging

PROCESS_LIFETIME_MINUTES = 5
QUEUE_TIMEOUT = 20

class EmbeddingErrorCode(str, Enum):
    BATCH_SIZE_EXCEEDED = "BATCH_SIZE_EXCEEDED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    TOKENIZATION_FAILED = "TOKENIZATION_FAILED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    TIMEOUT = "TIMEOUT"

class EmbeddingErrorResponse(BaseModel):
    code: EmbeddingErrorCode
    message: str
    details: Dict[str, Any] = {}

class EmbeddingError(Exception):
    _status_codes = {
        EmbeddingErrorCode.BATCH_SIZE_EXCEEDED: 422,
        EmbeddingErrorCode.MODEL_LOAD_FAILED: 500,
        EmbeddingErrorCode.TOKENIZATION_FAILED: 400,
        EmbeddingErrorCode.INFERENCE_FAILED: 500,
        EmbeddingErrorCode.TIMEOUT: 408,
    }

    def __init__(
        self, 
        code: EmbeddingErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def to_http_exception(self) -> HTTPException:
        return HTTPException(
            status_code=self._status_codes[self.code],
            detail=EmbeddingErrorResponse(
                code=self.code,
                message=self.message,
                details=self.details
            ).model_dump()
        )
    
    @classmethod
    def from_response(cls, response: EmbeddingErrorResponse) -> "EmbeddingError":
        return cls(
            code=response.code,
            message=response.message,
            details=response.details
        )

class Document(BaseModel):
    content: str

class EmbeddingRequest(BaseModel):
    model: str
    documents: List[Document]
    options: Optional[dict] = None

class EmbeddingResponse(BaseModel):
    model: str
    embeddings: List[List[float]]

class ModelConfig(BaseModel):
    max_length: int
    batch_size: int

    @classmethod
    def from_json(cls, json_path: str) -> ModelConfig:
        with open(json_path, "r") as f:
            return cls.model_validate_json(f.read())

def model_worker(
    model_id: str,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    shutdown_event: mp.Event,
    lifetime: int,
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if "/" not in model_id:
            raise EmbeddingError(
                code=EmbeddingErrorCode.MODEL_LOAD_FAILED,
                message="Invalid model ID format, expected 'model_org/model_name'",
            )

        _, model_name = model_id.split("/")
        config = ModelConfig.from_json(f"{model_name}.mlpackage/ember_config.json")

        if not model_name.endswith(".mlmodelc"):
            model_name += ".mlmodelc"
        
        try:
            model = ct.models.CompiledMLModel(model_name, compute_units=ct.ComputeUnit.CPU_ONLY)
        except Exception as e:
            raise EmbeddingError(
                code=EmbeddingErrorCode.MODEL_LOAD_FAILED,
                message=f"Failed to load model: {str(e)}",
                details={"model_name": model_name}
            )

        logger.info(f"Worker process for {model_id} started")
        logger.info(f"Model maximum sequence length: {config.max_length}")

        start_time = datetime.now()

        while not shutdown_event.is_set():
            if datetime.now() - start_time > timedelta(minutes=lifetime):
                logger.info(f"Worker for {model_id} reached lifetime limit")
                break

            try:
                input_data = input_queue.get(timeout=QUEUE_TIMEOUT)
                documents = input_data["documents"]

                if len(documents) > config.batch_size:
                    output_queue.put({
                        "success": False,
                        "error": EmbeddingErrorResponse(
                            code=EmbeddingErrorCode.BATCH_SIZE_EXCEEDED,
                            message=f"Batch size {len(documents)} exceeds maximum allowed size of {config.batch_size}",
                            details={
                                "max_batch_size": config.batch_size,
                                "received_size": len(documents)
                            }
                        ).model_dump()
                    })
                    continue

                try:
                    batch = tokenizer.batch_encode_plus(
                        documents,
                        padding="max_length",
                        max_length=config.max_length,
                        truncation=True,
                    )
                except Exception as e:
                    output_queue.put({
                        "success": False,
                        "error": EmbeddingErrorResponse(
                            code=EmbeddingErrorCode.TOKENIZATION_FAILED,
                            message=str(e),
                            details={"tokenizer": model_id}
                        ).model_dump()
                    })
                    continue

                input_ids = np.array(batch["input_ids"], dtype=np.int32)
                attention_mask = np.array(batch["attention_mask"], dtype=np.int32)
                current_batch_sz = input_ids.shape[0]

                if current_batch_sz < config.batch_size:
                    padder = lambda x: np.pad(
                        x,
                        ((0, config.batch_size - current_batch_sz), (0, 0)),
                        "constant",
                        constant_values=0,
                    )
                    input_ids = padder(input_ids)
                    attention_mask = padder(attention_mask)

                try:
                    prediction = model.predict(
                        {"input_ids": input_ids, "attention_mask": attention_mask}
                    )
                except Exception as e:
                    output_queue.put({
                        "success": False,
                        "error": EmbeddingErrorResponse(
                            code=EmbeddingErrorCode.INFERENCE_FAILED,
                            message=str(e),
                            details={"model_id": model_id}
                        ).model_dump()
                    })
                    continue

                embeddings = prediction["sentence_embeddings"][
                    :current_batch_sz
                ].tolist()
                output_queue.put({"success": True, "embeddings": embeddings})

            except queue.Empty:
                continue

    except EmbeddingError as e:
        output_queue.put({
            "success": False,
            "error": EmbeddingErrorResponse(
                code=e.code,
                message=e.message,
                details=e.details
            ).model_dump()
        })
    except Exception as e:
        output_queue.put({
            "success": False,
            "error": EmbeddingErrorResponse(
                code=EmbeddingErrorCode.MODEL_LOAD_FAILED,
                message=str(e),
                details={}
            ).model_dump()
        })
    finally:
        logger.info(f"Worker process for {model_id} shutting down")

class ProcessManager:
    def __init__(self):
        self.processes: Dict[str, Dict[str, Any]] = {}
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_expired_processes, daemon=True
        )
        self.cleanup_thread.start()

    def get_or_create_process(
        self, model_name: str, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get existing process or create new one if needed."""
        if (
            model_name not in self.processes
            or not self.processes[model_name]["process"].is_alive()
        ):
            if model_name in self.processes:
                self.cleanup_process(model_name)

            input_queue = mp.Queue()
            output_queue = mp.Queue()
            shutdown_event = mp.Event()

            process = mp.Process(
                target=model_worker,
                args=(
                    model_name,
                    input_queue,
                    output_queue,
                    shutdown_event,
                    options.get("keep_alive", PROCESS_LIFETIME_MINUTES),
                ),
            )
            process.start()

            self.processes[model_name] = {
                "process": process,
                "input_queue": input_queue,
                "output_queue": output_queue,
                "shutdown_event": shutdown_event,
                "start_time": datetime.now(),
                "keep_alive": options.get("keep_alive", PROCESS_LIFETIME_MINUTES),
            }

        return self.processes[model_name]

    def cleanup_process(self, model_name: str):
        """Clean up a specific process."""
        if model_name in self.processes:
            process_info = self.processes[model_name]
            process_info["shutdown_event"].set()
            process_info["process"].join(timeout=5)
            if process_info["process"].is_alive():
                process_info["process"].terminate()
            del self.processes[model_name]

    def _cleanup_expired_processes(self):
        """Background thread to clean up expired processes."""
        while True:
            current_time = datetime.now()
            for model_name in list(self.processes.keys()):
                process_info = self.processes[model_name]
                if current_time - process_info["start_time"] > timedelta(
                    minutes=process_info["keep_alive"]
                ):
                    logger.info(f"Cleaning up expired process for {model_name}")
                    self.cleanup_process(model_name)
            time.sleep(30)

app = FastAPI(title="CoreML Sentence Transformers API")
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()

logging.captureWarnings(True)
warning_logger = logging.getLogger("py.warnings")
warning_logger.handlers.clear()
warning_logger.addHandler(handler)

handler.setFormatter(DefaultFormatter("%(levelprefix)s %(message)s", use_colors=True))
logger.addHandler(handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

process_manager = ProcessManager()

@app.post("/api/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """Create embeddings for the given messages using the specified model."""
    try:
        process_info = process_manager.get_or_create_process(
            request.model, request.options if request.options else {}
        )

        process_info["input_queue"].put(
            {"documents": [doc.content for doc in request.documents]}
        )

        try:
            result = process_info["output_queue"].get(timeout=QUEUE_TIMEOUT)
        except queue.Empty:
            raise EmbeddingError(
                code=EmbeddingErrorCode.TIMEOUT,
                message="Request timed out while waiting for model response",
                details={"timeout_seconds": QUEUE_TIMEOUT}
            )

        if not result["success"]:
            error = EmbeddingError.from_response(
                EmbeddingErrorResponse.model_validate(result["error"])
            )
            raise error.to_http_exception()

        return EmbeddingResponse(model=request.model, embeddings=result["embeddings"])

    except EmbeddingError as e:
        raise e.to_http_exception()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.exception("Unexpected error in embedding endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List all available models in the registry."""
    return {"models": list(process_manager.processes.keys())}

def serve(host: str = "0.0.0.0", port: int = 11434):
    config = uvicorn.Config(
        app, 
        host=host, 
        port=port, 
        workers=1, 
        loop="asyncio", 
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    serve()
