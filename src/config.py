"""Central configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LiteLLM - set the primary model you want to use.
    # Examples:
    #   LITELLM_MODEL=groq/llama-3.1-8b-instant        + GROQ_API_KEY
    #   LITELLM_MODEL=gemini/gemini-2.5-flash          + GOOGLE_API_KEY
    #   LITELLM_MODEL=gpt-3.5-turbo                    + OPENAI_API_KEY
    #   LITELLM_MODEL=mistral/mistral-small            + MISTRAL_API_KEY
    #   LITELLM_MODEL=cohere/command-r                 + COHERE_API_KEY
    litellm_model: str = "groq/llama-3.1-8b-instant"

    # API keys - fill the one for your primary model, plus optional fallbacks.
    openai_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    mistral_api_key: str = ""
    cohere_api_key: str = ""
    anthropic_api_key: str = ""

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "clinical-triage-pipeline"

    # Paths
    data_raw_path: str = "data/sample/mtsamples.csv"
    delta_base_path: str = "data/delta"
    chroma_persist_dir: str = "data/chroma_db"
    use_vector_retrieval: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    warmup_on_startup: bool = False

    # Risk thresholds
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.4

    # Drift monitoring
    drift_alert_threshold: float = 0.05
    monitor_interval_hours: int = 6

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
