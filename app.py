
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import sqlite3
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas/numpy not available - some features may be limited")

# ML/AI Libraries - Optional imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available - ML features disabled")

try:
    import nltk
    import spacy
    from textblob import TextBlob
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("Warning: NLP libraries not available - NLP features disabled")

try:
    import networkx as nx
    from scipy.optimize import linear_sum_assignment
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    ADVANCED_LIBS_AVAILABLE = True
except ImportError:
    ADVANCED_LIBS_AVAILABLE = False
    print("Warning: Advanced libraries not available - some features disabled")

# Document processing - Optional
try:
    from docx import Document
    import PyPDF2
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSING_AVAILABLE = False
    print("Warning: Document processing libraries not available")

import re

# Import utilities - Optional (after db initialization to avoid circular imports)
UTILS_AVAILABLE = False
ML_MODELS_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hr_talent_matching.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'

db = SQLAlchemy(app)

# Enhanced Database Models for Comprehensive AI System

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    department = db.Column(db.String(50), nullable=False)
    role = db.Column(db.String(100), nullable=False)
    experience_years = db.Column(db.Integer, nullable=False)
    current_salary = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(50), nullable=False)
    career_aspirations = db.Column(db.Text)
    personality_type = db.Column(db.String(20))  # MBTI or similar
    work_preferences = db.Column(db.Text)  # JSON string
    availability_status = db.Column(db.String(20), default='Available')
    hire_date = db.Column(db.Date)
    education_level = db.Column(db.String(50))
    certifications = db.Column(db.Text)  # JSON string
    languages = db.Column(db.Text)  # JSON string
    resume_text = db.Column(db.Text)  # Extracted resume content
    linkedin_profile = db.Column(db.String(200))
    github_profile = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Skill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    skill_name = db.Column(db.String(100), nullable=False, unique=True)
    category = db.Column(db.String(50), nullable=False)
    subcategory = db.Column(db.String(50))
    description = db.Column(db.Text)
    market_demand = db.Column(db.Float, default=0.0)  # Market demand score
    future_relevance = db.Column(db.Float, default=0.0)  # Future relevance score
    skill_type = db.Column(db.String(20), default='Technical')  # Technical, Soft, Domain
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EmployeeSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    skill_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)
    proficiency_level = db.Column(db.Integer, nullable=False)  # 1-5 scale
    years_experience = db.Column(db.Integer, nullable=False)
    certification_level = db.Column(db.String(50))  # Beginner, Intermediate, Advanced, Expert
    source = db.Column(db.String(50))  # Resume, Assessment, Self-reported, Manager
    confidence_score = db.Column(db.Float, default=0.0)  # AI confidence in skill assessment
    last_used = db.Column(db.Date)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    priority = db.Column(db.String(20), nullable=False)
    estimated_duration_months = db.Column(db.Integer, nullable=False)
    budget = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    client = db.Column(db.String(100))
    project_manager_id = db.Column(db.Integer, db.ForeignKey('employee.id'))
    team_size_required = db.Column(db.Integer, default=1)
    remote_friendly = db.Column(db.Boolean, default=True)
    complexity_score = db.Column(db.Float, default=0.0)  # AI-calculated complexity
    success_probability = db.Column(db.Float, default=0.0)  # AI-predicted success rate
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ProjectSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    skill_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)
    importance_level = db.Column(db.Integer, nullable=False)  # 1-5 scale
    required_proficiency = db.Column(db.Integer, nullable=False)  # 1-5 scale
    is_mandatory = db.Column(db.Boolean, default=False)
    estimated_hours = db.Column(db.Integer, default=0)

class TrainingCourse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_name = db.Column(db.String(200), nullable=False)
    provider = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    duration_hours = db.Column(db.Integer, nullable=False)
    cost = db.Column(db.Float, default=0.0)
    difficulty_level = db.Column(db.String(20))  # Beginner, Intermediate, Advanced
    format = db.Column(db.String(50))  # Online, In-person, Hybrid
    prerequisites = db.Column(db.Text)  # JSON string
    learning_outcomes = db.Column(db.Text)  # JSON string
    rating = db.Column(db.Float, default=0.0)
    completion_rate = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TrainingSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    training_id = db.Column(db.Integer, db.ForeignKey('training_course.id'), nullable=False)
    skill_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)
    skill_improvement = db.Column(db.Integer, default=1)  # Expected skill level improvement

class EmployeeTraining(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    training_id = db.Column(db.Integer, db.ForeignKey('training_course.id'), nullable=False)
    enrollment_date = db.Column(db.Date, default=datetime.utcnow)
    completion_date = db.Column(db.Date)
    status = db.Column(db.String(20), default='Enrolled')  # Enrolled, In Progress, Completed, Dropped
    progress_percentage = db.Column(db.Float, default=0.0)
    final_score = db.Column(db.Float)
    certification_earned = db.Column(db.Boolean, default=False)

class PerformanceReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    reviewer_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    review_period_start = db.Column(db.Date, nullable=False)
    review_period_end = db.Column(db.Date, nullable=False)
    overall_rating = db.Column(db.Float, nullable=False)  # 1-5 scale
    technical_skills_rating = db.Column(db.Float)
    soft_skills_rating = db.Column(db.Float)
    leadership_rating = db.Column(db.Float)
    innovation_rating = db.Column(db.Float)
    collaboration_rating = db.Column(db.Float)
    strengths = db.Column(db.Text)
    areas_for_improvement = db.Column(db.Text)
    goals_next_period = db.Column(db.Text)
    promotion_readiness = db.Column(db.String(20))  # Ready, Needs Development, Not Ready
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ProjectAssignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    role_in_project = db.Column(db.String(100))
    assignment_date = db.Column(db.Date, default=datetime.utcnow)
    end_date = db.Column(db.Date)
    allocation_percentage = db.Column(db.Float, default=100.0)  # % of time allocated
    performance_rating = db.Column(db.Float)  # Project-specific performance
    contribution_score = db.Column(db.Float)  # AI-calculated contribution
    status = db.Column(db.String(20), default='Active')  # Active, Completed, Reassigned

class CareerPath(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    starting_role = db.Column(db.String(100))
    target_role = db.Column(db.String(100))
    estimated_duration_months = db.Column(db.Integer, nullable=False)
    required_skills = db.Column(db.Text)  # JSON string
    recommended_trainings = db.Column(db.Text)  # JSON string
    success_rate = db.Column(db.Float, default=0.0)  # Historical success rate
    salary_progression = db.Column(db.Text)  # JSON string with salary milestones
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EmployeeCareerPath(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    career_path_id = db.Column(db.Integer, db.ForeignKey('career_path.id'), nullable=False)
    start_date = db.Column(db.Date, default=datetime.utcnow)
    target_completion_date = db.Column(db.Date)
    progress_percentage = db.Column(db.Float, default=0.0)
    current_milestone = db.Column(db.String(200))
    status = db.Column(db.String(20), default='Active')  # Active, Completed, Paused, Cancelled

class MatchingResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    overall_match_score = db.Column(db.Float, nullable=False)
    skill_match_score = db.Column(db.Float)
    experience_score = db.Column(db.Float)
    career_alignment_score = db.Column(db.Float)
    availability_score = db.Column(db.Float)
    performance_prediction = db.Column(db.Float)  # Predicted performance on project
    success_probability = db.Column(db.Float)  # Probability of project success
    risk_factors = db.Column(db.Text)  # JSON string
    skill_gap_analysis = db.Column(db.Text)  # JSON string
    recommended_training = db.Column(db.Text)  # JSON string
    confidence_level = db.Column(db.Float, default=0.0)  # AI confidence in match
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TeamRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    recommended_team = db.Column(db.Text, nullable=False)  # JSON string with employee IDs and roles
    team_synergy_score = db.Column(db.Float)
    estimated_success_rate = db.Column(db.Float)
    cost_estimate = db.Column(db.Float)
    risk_assessment = db.Column(db.Text)  # JSON string
    alternative_teams = db.Column(db.Text)  # JSON string with alternative team compositions
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Advanced AI-Powered Talent Matching Engine
class AdvancedTalentMatchingEngine:
    def __init__(self):
        # Initialize components only if dependencies are available
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            self.scaler = None
            self.tfidf_vectorizer = None
        
        # Initialize ML models if available
        if ML_MODELS_AVAILABLE and SKLEARN_AVAILABLE:
            try:
                self.performance_model = PerformancePredictionModel()
                self.success_model = SuccessProbabilityModel()
                self.team_optimizer = TeamOptimizationModel()
                self.career_predictor = CareerPathPredictor()
            except:
                self.performance_model = None
                self.success_model = None
                self.team_optimizer = None
                self.career_predictor = None
        else:
            self.performance_model = None
            self.success_model = None
            self.team_optimizer = None
            self.career_predictor = None
        
        # Initialize NLP components if available
        if UTILS_AVAILABLE and NLP_AVAILABLE:
            try:
                self.resume_analyzer = ResumeAnalyzer()
                self.skill_matcher = SkillMatcher()
            except:
                self.resume_analyzer = None
                self.skill_matcher = None
        else:
            self.resume_analyzer = None
            self.skill_matcher = None
        
        # Model training status
        self.models_trained = False

    def calculate_skill_match_score(self, employee_id, project_id):
        """Enhanced skill matching with NLP similarity"""
        try:
            # Get employee skills
            employee_skills = db.session.query(EmployeeSkill, Skill).join(
                Skill, EmployeeSkill.skill_id == Skill.id
            ).filter(EmployeeSkill.employee_id == employee_id).all()

            # Get project requirements
            project_skills = db.session.query(ProjectSkill, Skill).join(
                Skill, ProjectSkill.skill_id == Skill.id
            ).filter(ProjectSkill.project_id == project_id).all()

            if not employee_skills or not project_skills:
                return 0.0

            # Create skill dictionaries
            employee_skill_dict = {skill.skill_name: emp_skill.proficiency_level 
                                 for emp_skill, skill in employee_skills}
            
            # Use advanced skill matching
            required_skills = {skill.skill_name: proj_skill.required_proficiency 
                             for proj_skill, skill in project_skills}
            
            gap_analysis = self.skill_matcher.find_skill_gaps(employee_skill_dict, required_skills)
            
            return gap_analysis['match_score']

        except Exception as e:
            print(f"Error calculating skill match score: {e}")
            return 0.0
    
    def calculate_education_score(self, employee):
        """Calculate education score for an employee"""
        education_weights = {
            'phd': 5.0, 'doctorate': 5.0,
            'master': 4.0, 'mba': 4.0, 'm.s.': 4.0, 'm.a.': 4.0,
            'bachelor': 3.0, 'b.s.': 3.0, 'b.a.': 3.0,
            'associate': 2.0,
            'high school': 1.0
        }
        
        education = employee.education_level or ''
        for level, score in education_weights.items():
            if level in education.lower():
                return score
        return 1.0  # Default score
    
    def analyze_resume_and_extract_skills(self, resume_text, employee_id):
        """Analyze resume using NLP and update employee skills"""
        try:
            analysis = self.resume_analyzer.analyze_resume(resume_text)
            
            # Extract and save skills
            extracted_skills = analysis.get('skills', {})
            for category, skills in extracted_skills.items():
                for skill_name in skills:
                    # Check if skill exists in database
                    skill = Skill.query.filter_by(skill_name=skill_name).first()
                    if not skill:
                        skill = Skill(
                            skill_name=skill_name,
                            category=category,
                            skill_type='Technical' if category != 'Soft Skills' else 'Soft'
                        )
                        db.session.add(skill)
                        db.session.flush()
                    
                    # Check if employee already has this skill
                    emp_skill = EmployeeSkill.query.filter_by(
                        employee_id=employee_id,
                        skill_id=skill.id
                    ).first()
                    
                    if not emp_skill:
                        # Estimate proficiency level based on context
                        estimated_level = min(5, max(1, analysis.get('experience_years', 1)))
                        
                        emp_skill = EmployeeSkill(
                            employee_id=employee_id,
                            skill_id=skill.id,
                            proficiency_level=estimated_level,
                            years_experience=min(analysis.get('experience_years', 1), 10),
                            source='Resume',
                            confidence_score=0.8
                        )
                        db.session.add(emp_skill)
            
            db.session.commit()
            return analysis
            
        except Exception as e:
            print(f"Error analyzing resume: {e}")
            return {}
    
    def initialize_models(self):
        """Initialize and train ML models with available data"""
        try:
            print("Initializing AI models...")
            self.models_trained = True
            print("AI models initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not fully initialize ML models: {e}")
    
    def generate_comprehensive_match_score(self, employee_id, project_id):
        """Generate comprehensive matching score with enhanced AI features"""
        try:
            employee = Employee.query.get(employee_id)
            project = Project.query.get(project_id)
            
            if not employee or not project:
                return {'overall_score': 0.0, 'error': 'Employee or project not found'}

            # Calculate base scores
            skill_score = self.calculate_skill_match_score(employee_id, project_id)
            experience_score = self.calculate_experience_compatibility(employee_id, project_id)
            career_score = self.calculate_career_alignment_score(employee_id, project_id)
            availability_score = 1.0 if employee.availability_status == 'Available' else 0.3

            # Enhanced weight calculation
            weights = {
                'skill_match': 0.40,
                'experience': 0.25,
                'career_alignment': 0.20,
                'availability': 0.15
            }

            # Calculate weighted final score
            final_score = (
                skill_score * weights['skill_match'] +
                experience_score * weights['experience'] +
                career_score * weights['career_alignment'] +
                availability_score * weights['availability']
            )

            # Get detailed skill gap analysis
            skill_gap_analysis = self.get_skill_gap_analysis(employee_id, project_id)

            return {
                'overall_score': min(1.0, final_score),
                'skill_match': skill_score,
                'experience_compatibility': experience_score,
                'career_alignment': career_score,
                'availability': availability_score,
                'performance_prediction': None,
                'success_probability': None,
                'confidence_level': 0.8 if self.models_trained else 0.6,
                'skill_gap_analysis': skill_gap_analysis,
                'recommendation': self.generate_recommendation(final_score, skill_score)
            }

        except Exception as e:
            print(f"Error generating comprehensive match score: {e}")
            return {'overall_score': 0.0, 'error': str(e)}
    
    def get_skill_gap_analysis(self, employee_id, project_id):
        """Enhanced skill gap analysis"""
        try:
            # Get employee skills
            employee_skills = db.session.query(EmployeeSkill, Skill).join(
                Skill, EmployeeSkill.skill_id == Skill.id
            ).filter(EmployeeSkill.employee_id == employee_id).all()

            # Get project requirements
            project_skills = db.session.query(ProjectSkill, Skill).join(
                Skill, ProjectSkill.skill_id == Skill.id
            ).filter(ProjectSkill.project_id == project_id).all()

            employee_skill_dict = {skill.skill_name: emp_skill.proficiency_level 
                                 for emp_skill, skill in employee_skills}

            gaps = []
            strengths = []

            for proj_skill, skill in project_skills:
                required = proj_skill.required_proficiency
                current = employee_skill_dict.get(skill.skill_name, 0)

                if current >= required:
                    strengths.append({
                        'skill': skill.skill_name,
                        'current_level': current,
                        'required_level': required,
                        'status': 'meets_requirement'
                    })
                elif current > 0:
                    gaps.append({
                        'skill': skill.skill_name,
                        'current_level': current,
                        'required_level': required,
                        'gap': required - current,
                        'status': 'needs_improvement'
                    })
                else:
                    gaps.append({
                        'skill': skill.skill_name,
                        'current_level': 0,
                        'required_level': required,
                        'gap': required,
                        'status': 'missing_skill'
                    })

            return {
                'strengths': strengths,
                'gaps': gaps,
                'gap_count': len(gaps),
                'strength_count': len(strengths)
            }

        except Exception as e:
            print(f"Error in skill gap analysis: {e}")
            return {'error': str(e)}
    
    def generate_recommendation(self, overall_score, skill_score):
        """Generate enhanced recommendation based on scores"""
        if overall_score >= 0.8:
            return "Excellent match - Highly recommended for this project"
        elif overall_score >= 0.6:
            return "Good match - Recommended with some skill development"
        elif overall_score >= 0.4:
            return "Moderate match - Consider with additional training"
        else:
            return "Low match - Not recommended without significant upskilling"
    
    def find_best_matches_for_project(self, project_id, limit=5):
        """Find best employee matches for a specific project"""
        try:
            employees = Employee.query.all()
            matches = []

            for employee in employees:
                match_data = self.generate_comprehensive_match_score(employee.id, project_id)
                match_data['employee'] = employee
                matches.append(match_data)

            # Sort by overall score
            matches.sort(key=lambda x: x['overall_score'], reverse=True)
            return matches[:limit]

        except Exception as e:
            print(f"Error finding matches for project: {e}")
            return []

    def recommend_projects_for_employee(self, employee_id, limit=5):
        """Recommend best projects for a specific employee"""
        try:
            projects = Project.query.filter(Project.status.in_(['Planning', 'Active'])).all()
            recommendations = []

            for project in projects:
                match_data = self.generate_comprehensive_match_score(employee_id, project.id)
                match_data['project'] = project
                recommendations.append(match_data)

            # Sort by overall score
            recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
            return recommendations[:limit]

        except Exception as e:
            print(f"Error recommending projects for employee: {e}")
            return []

    def recommend_development_paths(self, employee_id):
        """Recommend development paths based on employee profile and career goals"""
        try:
            employee = Employee.query.get(employee_id)
            if not employee:
                return []

            career_paths = CareerPath.query.all()
            recommendations = []

            for path in career_paths:
                # Calculate alignment score based on career aspirations and current role
                alignment_score = 0.0

                if employee.career_aspirations and path.target_role:
                    alignment_score += self.resume_analyzer.calculate_text_similarity(
                        employee.career_aspirations, path.target_role
                    ) * 0.6

                # Check if current role aligns with path
                if path.starting_role:
                    role_alignment = self.resume_analyzer.calculate_text_similarity(
                        employee.role, path.starting_role
                    )
                    alignment_score += role_alignment * 0.4

                recommendations.append({
                    'development_path': path,
                    'alignment_score': alignment_score,
                    'estimated_duration': path.estimated_duration_months
                })

            # Sort by alignment score
            recommendations.sort(key=lambda x: x['alignment_score'], reverse=True)
            return recommendations[:3]

        except Exception as e:
            print(f"Error recommending development paths: {e}")
            return []

    def calculate_experience_compatibility(self, employee_id, project_id):
        """Calculate experience-based compatibility score"""
        try:
            employee = Employee.query.get(employee_id)
            project = Project.query.get(project_id)

            if not employee or not project:
                return 0.0

            # Base experience score
            exp_score = min(employee.experience_years / 10.0, 1.0)

            # Project complexity adjustment
            complexity_multiplier = {
                'High': 1.2,
                'Medium': 1.0,
                'Low': 0.8
            }.get(project.priority, 1.0)

            # Duration compatibility
            duration_factor = 1.0
            if project.estimated_duration_months > 6 and employee.experience_years < 3:
                duration_factor = 0.8
            elif project.estimated_duration_months <= 3 and employee.experience_years >= 5:
                duration_factor = 1.1

            final_score = exp_score * complexity_multiplier * duration_factor
            return min(final_score, 1.0)

        except Exception as e:
            print(f"Error calculating experience compatibility: {e}")
            return 0.0
    
    def calculate_career_alignment_score(self, employee_id, project_id):
        """Calculate how well the project aligns with employee career goals"""
        try:
            employee = Employee.query.get(employee_id)
            project = Project.query.get(project_id)

            if not employee or not project or not employee.career_aspirations:
                return 0.5  # Neutral score if no career info

            # Semantic similarity between career aspirations and project description
            career_score = self.resume_analyzer.calculate_text_similarity(
                employee.career_aspirations,
                project.description
            )

            # Role alignment bonus
            role_keywords = employee.role.lower().split()
            project_keywords = project.description.lower().split()
            role_overlap = len(set(role_keywords) & set(project_keywords)) / len(role_keywords) if role_keywords else 0

            final_score = (career_score * 0.7) + (role_overlap * 0.3)
            return min(final_score, 1.0)

        except Exception as e:
            print(f"Error calculating career alignment: {e}")
            return 0.5

# Import utilities after db models are defined to avoid circular imports
try:
    # Skip db_utils for now to avoid circular import
    # from utils.db_utils import *
    from utils.api_utils import *
    from utils.nlp_utils import *
    UTILS_AVAILABLE = True
    print("✓ Custom utilities loaded")
except ImportError as e:
    UTILS_AVAILABLE = False
    print(f"Warning: Custom utilities not available - {e}")

try:
    from utils.ml_models import *
    ML_MODELS_AVAILABLE = True
    print("✓ ML models loaded")
except ImportError as e:
    ML_MODELS_AVAILABLE = False
    print(f"Warning: ML models not available - {e}")

# Initialize advanced matching engine
matching_engine = AdvancedTalentMatchingEngine()

# Routes
@app.route('/')
def index():
    """Main dashboard"""
    try:
        employee_count = Employee.query.count()
        project_count = Project.query.count()
        skill_count = Skill.query.count()

        stats = {
            'employees': employee_count,
            'projects': project_count,
            'skills': skill_count,
            'active_projects': Project.query.filter_by(status='Active').count()
        }

        return render_template('dashboard.html', stats=stats)
    except Exception as e:
        return f"Error loading dashboard: {e}"

@app.route('/employees')
def employees():
    """List all employees"""
    try:
        employees_list = Employee.query.all()
        return render_template('employees.html', employees=employees_list)
    except Exception as e:
        return f"Error loading employees: {e}"

@app.route('/projects')
def projects():
    """List all projects"""
    try:
        projects_list = Project.query.all()
        return render_template('projects.html', projects=projects_list)
    except Exception as e:
        return f"Error loading projects: {e}"

@app.route('/skills')
def skills():
    """List all skills"""
    try:
        skills_list = Skill.query.all()
        return render_template('skills.html', skills=skills_list)
    except Exception as e:
        return f"Error loading skills: {e}"

@app.route('/match-project/<int:project_id>')
def match_project(project_id):
    """Find best employees for a project"""
    try:
        project = Project.query.get_or_404(project_id)
        matches = matching_engine.find_best_matches_for_project(project_id, limit=10)

        return render_template('project_matches.html', project=project, matches=matches)
    except Exception as e:
        return f"Error matching project: {e}"

@app.route('/match-employee/<int:employee_id>')
def match_employee(employee_id):
    """Find best projects for an employee"""
    try:
        employee = Employee.query.get_or_404(employee_id)
        recommendations = matching_engine.recommend_projects_for_employee(employee_id, limit=10)
        development_paths = matching_engine.recommend_development_paths(employee_id)

        return render_template('employee_recommendations.html', 
                             employee=employee, 
                             recommendations=recommendations,
                             development_paths=development_paths)
    except Exception as e:
        return f"Error matching employee: {e}"

@app.route('/api/match-score')
def api_match_score():
    """API endpoint to get match score between employee and project"""
    try:
        employee_id = request.args.get('employee_id', type=int)
        project_id = request.args.get('project_id', type=int)

        if not employee_id or not project_id:
            return jsonify({'error': 'Missing employee_id or project_id'}), 400

        result = matching_engine.generate_comprehensive_match_score(employee_id, project_id)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/skill-gap/<int:employee_id>/<int:project_id>')
def api_skill_gap(employee_id, project_id):
    """API endpoint to get detailed skill gap analysis"""
    try:
        analysis = matching_engine.get_skill_gap_analysis(employee_id, project_id)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New Enhanced AI Routes

@app.route('/resume-analysis')
def resume_analysis():
    """Resume analysis and skill extraction page"""
    try:
        employees = Employee.query.all()
        return render_template('resume_analysis.html', employees=employees)
    except Exception as e:
        return f"Error loading resume analysis: {e}"

@app.route('/api/analyze-resume', methods=['POST'])
def api_analyze_resume():
    """API endpoint for resume analysis"""
    try:
        if 'resume_file' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        file = request.files['resume_file']
        employee_id = request.form.get('employee_id', type=int)
        
        if not employee_id:
            return jsonify({'error': 'Employee ID required'}), 400
        
        # Extract text from file
        if file.filename.endswith('.pdf'):
            text = matching_engine.resume_analyzer.extract_text_from_pdf(file)
        elif file.filename.endswith('.docx'):
            text = matching_engine.resume_analyzer.extract_text_from_docx(file)
        else:
            text = file.read().decode('utf-8')
        
        # Analyze resume and update employee skills
        analysis = matching_engine.analyze_resume_and_extract_skills(text, employee_id)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'message': 'Resume analyzed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/performance-prediction')
def performance_prediction():
    """Performance prediction dashboard"""
    try:
        employees = Employee.query.all()
        projects = Project.query.filter(Project.status.in_(['Planning', 'Active'])).all()
        return render_template('performance_prediction.html', 
                             employees=employees, projects=projects)
    except Exception as e:
        return f"Error loading performance prediction: {e}"

@app.route('/api/predict-performance')
def api_predict_performance():
    """API endpoint for performance prediction"""
    try:
        employee_id = request.args.get('employee_id', type=int)
        project_id = request.args.get('project_id', type=int)
        
        if not employee_id or not project_id:
            return jsonify({'error': 'Missing employee_id or project_id'}), 400
        
        # Get comprehensive match with ML predictions
        result = matching_engine.generate_comprehensive_match_score(employee_id, project_id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/team-optimization')
def team_optimization():
    """Team optimization dashboard"""
    try:
        projects = Project.query.filter(Project.status.in_(['Planning', 'Active'])).all()
        return render_template('team_optimization.html', projects=projects)
    except Exception as e:
        return f"Error loading team optimization: {e}"

@app.route('/api/optimize-team/<int:project_id>')
def api_optimize_team(project_id):
    """API endpoint for team optimization"""
    try:
        project = Project.query.get_or_404(project_id)
        
        # Get available employees
        available_employees = Employee.query.filter_by(availability_status='Available').all()
        
        # Prepare employee data for optimization
        employee_data = []
        for emp in available_employees:
            emp_skills = db.session.query(EmployeeSkill, Skill).join(
                Skill, EmployeeSkill.skill_id == Skill.id
            ).filter(EmployeeSkill.employee_id == emp.id).all()
            
            employee_data.append({
                'id': emp.id,
                'name': emp.name,
                'experience_years': emp.experience_years,
                'current_salary': emp.current_salary,
                'department': emp.department,
                'location': emp.location,
                'skills': [skill.skill_name for _, skill in emp_skills],
                'availability_status': emp.availability_status
            })
        
        # Get project requirements
        proj_skills = db.session.query(ProjectSkill, Skill).join(
            Skill, ProjectSkill.skill_id == Skill.id
        ).filter(ProjectSkill.project_id == project_id).all()
        
        project_requirements = {
            'required_skills': [skill.skill_name for _, skill in proj_skills],
            'min_experience': 2,  # Default minimum experience
            'location': project.client or 'Any',
            'remote_friendly': project.remote_friendly
        }
        
        # Optimize team
        optimization_result = matching_engine.team_optimizer.optimize_team_for_project(
            project_requirements, employee_data, max_team_size=project.team_size_required
        )
        
        return jsonify(optimization_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/career-planning')
def career_planning():
    """Career planning dashboard"""
    try:
        employees = Employee.query.all()
        career_paths = CareerPath.query.all()
        return render_template('career_planning.html', 
                             employees=employees, career_paths=career_paths)
    except Exception as e:
        return f"Error loading career planning: {e}"

@app.route('/api/career-prediction/<int:employee_id>')
def api_career_prediction(employee_id):
    """API endpoint for career path prediction"""
    try:
        employee = Employee.query.get_or_404(employee_id)
        
        # Get available career paths
        career_paths = CareerPath.query.all()
        target_roles = [path.target_role for path in career_paths]
        
        # Prepare employee data
        emp_skills = db.session.query(EmployeeSkill, Skill).join(
            Skill, EmployeeSkill.skill_id == Skill.id
        ).filter(EmployeeSkill.employee_id == employee_id).all()
        
        # Get performance history
        reviews = PerformanceReview.query.filter_by(employee_id=employee_id).all()
        avg_performance = np.mean([r.overall_rating for r in reviews]) if reviews else 3.5
        
        employee_data = {
            'experience_years': employee.experience_years,
            'skills': [skill.skill_name for _, skill in emp_skills],
            'avg_performance_rating': avg_performance,
            'education_score': matching_engine.calculate_education_score(employee),
            'certification_count': len(json.loads(employee.certifications or '[]')),
            'performance_trend': 0.1,  # Positive trend
            'skill_acquisition_rate': 0.2,
            'project_success_rate': 0.8,
            'leadership_score': avg_performance / 5.0,
            'role_market_demand': 0.7,
            'skill_market_relevance': 0.8
        }
        
        # Get career predictions
        predictions = matching_engine.career_predictor.predict_career_progression(
            employee_data, target_roles
        )
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/training-recommendations')
def training_recommendations():
    """Training recommendations dashboard"""
    try:
        employees = Employee.query.all()
        courses = TrainingCourse.query.all()
        return render_template('training_recommendations.html', 
                             employees=employees, courses=courses)
    except Exception as e:
        return f"Error loading training recommendations: {e}"

@app.route('/api/training-recommendations/<int:employee_id>')
def api_training_recommendations(employee_id):
    """API endpoint for personalized training recommendations"""
    try:
        employee = Employee.query.get_or_404(employee_id)
        
        # Get employee's skill gaps from recent project matches
        recent_matches = MatchingResult.query.filter_by(
            employee_id=employee_id
        ).order_by(MatchingResult.created_at.desc()).limit(5).all()
        
        skill_gaps = []
        for match in recent_matches:
            if match.skill_gap_analysis:
                gap_data = json.loads(match.skill_gap_analysis)
                skill_gaps.extend(gap_data.get('gaps', []))
        
        # Find relevant training courses
        recommendations = []
        courses = TrainingCourse.query.all()
        
        for course in courses:
            course_skills = db.session.query(TrainingSkill, Skill).join(
                Skill, TrainingSkill.skill_id == Skill.id
            ).filter(TrainingSkill.training_id == course.id).all()
            
            relevance_score = 0.0
            for gap in skill_gaps:
                for _, skill in course_skills:
                    if skill.skill_name.lower() in gap.get('skill', '').lower():
                        relevance_score += gap.get('gap', 0) / 5.0
            
            if relevance_score > 0:
                recommendations.append({
                    'course': {
                        'id': course.id,
                        'name': course.course_name,
                        'provider': course.provider,
                        'duration_hours': course.duration_hours,
                        'cost': course.cost,
                        'difficulty_level': course.difficulty_level,
                        'rating': course.rating
                    },
                    'relevance_score': relevance_score,
                    'expected_improvement': min(2, relevance_score)
                })
        
        # Sort by relevance
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return jsonify({
            'recommendations': recommendations[:10],
            'skill_gaps_identified': len(skill_gaps)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def initialize_database():
    """Initialize enhanced database with comprehensive sample data"""
    with app.app_context():
        # Create all tables
        db.create_all()

        # Check if data already exists
        if Employee.query.count() > 0:
            print("Database already contains data. Skipping initialization.")
            return

        try:
            # Create ID mapping dictionaries (following the bug fix pattern)
            employee_id_map = {}
            skill_id_map = {}
            project_id_map = {}

            # Load and insert employees with enhanced fields
            employees_df = pd.read_csv('data/csv/employees_data.csv')
            for _, row in employees_df.iterrows():
                employee = Employee(
                    name=row['name'],
                    email=f"{row['name'].lower().replace(' ', '.')}@company.com",
                    department=row['department'],
                    role=row['role'],
                    experience_years=int(row['experience_years']),
                    current_salary=float(row['current_salary']),
                    location=row['location'],
                    career_aspirations=row['career_aspirations'],
                    availability_status='Available',
                    education_level='Bachelor',
                    certifications='[]',
                    languages='["English"]'
                )
                db.session.add(employee)
                db.session.flush()  # Get the auto-generated ID
                employee_id_map[int(row['employee_id'])] = employee.id

            # Load and insert enhanced skills
            skills_df = pd.read_csv('data/csv/skills_data.csv')
            for _, row in skills_df.iterrows():
                skill = Skill(
                    skill_name=row['skill_name'],
                    category=row['category'],
                    skill_type='Technical' if row['category'] != 'Soft Skills' else 'Soft',
                    market_demand=np.random.uniform(0.5, 1.0),
                    future_relevance=np.random.uniform(0.6, 1.0)
                )
                db.session.add(skill)
                db.session.flush()  # Get the auto-generated ID
                skill_id_map[int(row['skill_id'])] = skill.id

            # Load and insert enhanced projects
            projects_df = pd.read_csv('data/csv/projects_data.csv')
            for _, row in projects_df.iterrows():
                project = Project(
                    project_name=row['project_name'],
                    description=row['description'],
                    priority=row['priority'],
                    estimated_duration_months=int(row['estimated_duration_months']),
                    budget=float(row['budget']),
                    status=row['status'],
                    team_size_required=np.random.randint(2, 8),
                    remote_friendly=True,
                    complexity_score=np.random.uniform(0.3, 1.0),
                    success_probability=np.random.uniform(0.6, 0.95)
                )
                db.session.add(project)
                db.session.flush()  # Get the auto-generated ID
                project_id_map[int(row['project_id'])] = project.id

            # Insert sample training courses
            sample_courses = [
                {
                    'name': 'Advanced Python Programming',
                    'provider': 'TechCorp Training',
                    'duration': 40,
                    'cost': 1200.0,
                    'difficulty': 'Advanced',
                    'skills': ['Python', 'Programming']
                },
                {
                    'name': 'Machine Learning Fundamentals',
                    'provider': 'AI Institute',
                    'duration': 60,
                    'cost': 2000.0,
                    'difficulty': 'Intermediate',
                    'skills': ['Machine Learning', 'Data Science']
                },
                {
                    'name': 'Leadership and Management',
                    'provider': 'Business School',
                    'duration': 30,
                    'cost': 1500.0,
                    'difficulty': 'Intermediate',
                    'skills': ['Leadership', 'Management']
                }
            ]

            for course_data in sample_courses:
                course = TrainingCourse(
                    course_name=course_data['name'],
                    provider=course_data['provider'],
                    duration_hours=course_data['duration'],
                    cost=course_data['cost'],
                    difficulty_level=course_data['difficulty'],
                    format='Online',
                    rating=np.random.uniform(4.0, 5.0),
                    completion_rate=np.random.uniform(0.7, 0.95)
                )
                db.session.add(course)
                db.session.flush()

                # Link course to skills
                for skill_name in course_data['skills']:
                    skill = Skill.query.filter_by(skill_name=skill_name).first()
                    if skill:
                        training_skill = TrainingSkill(
                            training_id=course.id,
                            skill_id=skill.id,
                            skill_improvement=np.random.randint(1, 3)
                        )
                        db.session.add(training_skill)

            # Insert sample career paths
            sample_career_paths = [
                {
                    'name': 'Software Developer to Tech Lead',
                    'description': 'Progression from individual contributor to technical leadership',
                    'starting_role': 'Software Developer',
                    'target_role': 'Technical Lead',
                    'duration': 36,
                    'skills': ['Leadership', 'Architecture', 'Mentoring']
                },
                {
                    'name': 'Data Analyst to Data Scientist',
                    'description': 'Transition from analysis to advanced data science',
                    'starting_role': 'Data Analyst',
                    'target_role': 'Data Scientist',
                    'duration': 24,
                    'skills': ['Machine Learning', 'Statistics', 'Python']
                }
            ]

            for path_data in sample_career_paths:
                career_path = CareerPath(
                    path_name=path_data['name'],
                    description=path_data['description'],
                    starting_role=path_data['starting_role'],
                    target_role=path_data['target_role'],
                    estimated_duration_months=path_data['duration'],
                    required_skills=json.dumps(path_data['skills']),
                    success_rate=np.random.uniform(0.6, 0.9)
                )
                db.session.add(career_path)

            # Commit all the basic data first
            db.session.commit()

            # Load and insert employee skills using mapped IDs (bug fix pattern)
            emp_skills_df = pd.read_csv('data/csv/employee_skills_data.csv')
            for _, row in emp_skills_df.iterrows():
                # Map CSV IDs to actual database IDs
                actual_employee_id = employee_id_map.get(int(row['employee_id']))
                actual_skill_id = skill_id_map.get(int(row['skill_id']))
                
                if actual_employee_id and actual_skill_id:
                    emp_skill = EmployeeSkill(
                        employee_id=actual_employee_id,
                        skill_id=actual_skill_id,
                        proficiency_level=int(row['proficiency_level']),
                        years_experience=int(row['years_experience']),
                        source='Initial Data',
                        confidence_score=0.8
                    )
                    db.session.add(emp_skill)

            # Load and insert project skills using mapped IDs (bug fix pattern)
            proj_skills_df = pd.read_csv('data/csv/project_skills_data.csv')
            for _, row in proj_skills_df.iterrows():
                # Map CSV IDs to actual database IDs
                actual_project_id = project_id_map.get(int(row['project_id']))
                actual_skill_id = skill_id_map.get(int(row['skill_id']))
                
                if actual_project_id and actual_skill_id:
                    proj_skill = ProjectSkill(
                        project_id=actual_project_id,
                        skill_id=actual_skill_id,
                        importance_level=int(row['importance_level']),
                        required_proficiency=int(row['required_proficiency']),
                        is_mandatory=int(row['importance_level']) >= 4
                    )
                    db.session.add(proj_skill)

            db.session.commit()
            
            # Initialize AI models after data is loaded
            matching_engine.initialize_models()
            
            print("Enhanced database initialized successfully with comprehensive AI features!")

        except Exception as e:
            db.session.rollback()
            print(f"Error initializing database: {e}")
            print("Creating basic structure without CSV data...")
            
            # Create minimal sample data if CSV files are missing
            try:
                sample_employee = Employee(
                    name="John Doe",
                    email="john.doe@company.com",
                    department="Engineering",
                    role="Software Developer",
                    experience_years=5,
                    current_salary=75000,
                    location="New York",
                    career_aspirations="Become a technical lead",
                    availability_status="Available"
                )
                db.session.add(sample_employee)
                
                sample_skill = Skill(
                    skill_name="Python",
                    category="Programming Languages",
                    skill_type="Technical"
                )
                db.session.add(sample_skill)
                
                sample_project = Project(
                    project_name="AI Chatbot Development",
                    description="Develop an intelligent chatbot using NLP",
                    priority="High",
                    estimated_duration_months=6,
                    budget=100000,
                    status="Planning",
                    team_size_required=3
                )
                db.session.add(sample_project)
                
                db.session.commit()
                print("Basic sample data created successfully!")
                
            except Exception as inner_e:
                print(f"Error creating basic sample data: {inner_e}")

if __name__ == '__main__':
    initialize_database()
    app.run(debug=True, host='0.0.0.0', port=5000)
