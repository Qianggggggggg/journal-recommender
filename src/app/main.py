"""FastAPI 应用入口"""
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .api import router
from ..utils.logging import setup_logging


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="论文投稿期刊推荐系统",
        description="根据论文内容推荐适合投稿的计算机类期刊",
        version="0.1.0",
    )

    # 注册路由
    app.include_router(router, prefix="/api")

    # 静态文件（前端）
    frontend_path = Path("frontend")
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory="frontend"), name="static")

    @app.get("/")
    async def root():
        """根路径重定向到前端"""
        frontend_index = Path("frontend/index.html")
        if frontend_index.exists():
            return RedirectResponse(url="/static/index.html")
        return {"message": "Journal Recommender API", "version": "0.1.0"}

    return app


app = create_app()


if __name__ == "__main__":
    import yaml

    # 加载配置
    config_path = Path("configs/app.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        host = config.get("app", {}).get("host", "0.0.0.0")
        port = config.get("app", {}).get("port", 8000)
        log_level = config.get("app", {}).get("log_level", "INFO")
    else:
        host = "0.0.0.0"
        port = 8000
        log_level = "INFO"

    setup_logging(level=log_level)

    uvicorn.run(app, host=host, port=port)