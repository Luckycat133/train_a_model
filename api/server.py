import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Generator
import torch
import uvicorn
import json

from generate import (
    load_model,
    load_tokenizer,
    GenerationConfig,
    generate_text,
    generate_stream,
    setup_logger
)

app = FastAPI(
    title="Lingmao Moyun Inference API",
    description="灵猫模型推理 API - 支持文本生成和流式输出",
    version="1.0.0"
)

model = None
tokenizer = None
device = None


class GenerateRequest(BaseModel):
    """生成请求模型"""
    prompt: Union[str, List[str]] = Field(..., description="输入提示文本或文本列表")
    max_length: Optional[int] = Field(100, description="最大生成长度")
    max_new_tokens: Optional[int] = Field(None, description="最大新生成 token 数")
    temperature: Optional[float] = Field(0.7, description="温度参数，控制随机性")
    top_k: Optional[int] = Field(50, description="Top-K 采样参数")
    top_p: Optional[float] = Field(0.9, description="Top-P (nucleus) 采样参数")
    repetition_penalty: Optional[float] = Field(1.0, description="重复惩罚参数")
    stream: Optional[bool] = Field(False, description="是否使用流式输出")


class GenerateResponse(BaseModel):
    """生成响应模型"""
    prompt: str
    text: str
    success: bool = True


class BatchGenerateResponse(BaseModel):
    """批量生成响应模型"""
    results: List[GenerateResponse]


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model, tokenizer, device
    
    setup_logger()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer_path = os.getenv("TOKENIZER_PATH", "tokenizer.json")
    model_path = os.getenv("MODEL_PATH", "model_weights/best_model_v0.4.pt")
    
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        raise RuntimeError("无法加载分词器")
    
    model = load_model(model_path, device=device)
    if model is None:
        raise RuntimeError("无法加载模型")
    
    print(f"模型已加载到设备: {device}")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Lingmao Moyun Inference API",
        "version": "1.0.0",
        "device": device,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }


@app.post("/generate", response_model=Union[GenerateResponse, BatchGenerateResponse])
async def generate(request: GenerateRequest):
    """生成文本接口
    
    支持单条和批量生成，不使用流式输出时使用此接口。
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        config = GenerationConfig(
            max_length=request.max_length,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
        
        if request.stream:
            raise HTTPException(status_code=400, detail="流式输出请使用 /generate/stream 接口")
        
        result = generate_text(model, tokenizer, request.prompt, config, device)
        
        if isinstance(result, str):
            return GenerateResponse(prompt=request.prompt, text=result)
        else:
            results = []
            for p, t in zip(request.prompt, result):
                results.append(GenerateResponse(prompt=p, text=t))
            return BatchGenerateResponse(results=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream")
async def generate_stream_endpoint(request: GenerateRequest):
    """流式生成文本接口
    
    使用 Server-Sent Events (SSE) 格式返回逐步生成的文本。
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    if isinstance(request.prompt, list):
        raise HTTPException(status_code=400, detail="流式生成不支持批量输入")
    
    try:
        config = GenerationConfig(
            max_length=request.max_length,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
        
        async def stream_generator():
            yield f"data: {json.dumps({'prompt': request.prompt}, ensure_ascii=False)}\n\n"
            
            for text in generate_stream(model, tokenizer, request.prompt, config, device):
                yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """获取模型信息"""
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="分词器未加载")
    
    return {
        "vocab_size": len(tokenizer.token_to_id) if hasattr(tokenizer, 'token_to_id') else None,
        "unk_token": tokenizer.unk_token if hasattr(tokenizer, 'unk_token') else None,
        "eos_token": tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else None,
        "device": device
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
