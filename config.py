import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///hr_talent_matching.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # AI Model Configuration
    MAX_FEATURES = 1000
    MIN_SIMILARITY_THRESHOLD = 0.3
    DEFAULT_BATCH_SIZE = 10

    # Matching Algorithm Weights
    SKILL_WEIGHT = 0.45
    EXPERIENCE_WEIGHT = 0.25
    CAREER_ALIGNMENT_WEIGHT = 0.20
    AVAILABILITY_WEIGHT = 0.10

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_ECHO = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
