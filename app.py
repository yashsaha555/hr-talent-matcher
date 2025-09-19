
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import sqlite3
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hr_talent_matching.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'

db = SQLAlchemy(app)

# Database Models
class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    role = db.Column(db.String(100), nullable=False)
    experience_years = db.Column(db.Integer, nullable=False)
    current_salary = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(50), nullable=False)
    career_aspirations = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Skill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    skill_name = db.Column(db.String(100), nullable=False, unique=True)
    category = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)

class EmployeeSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    skill_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)
    proficiency_level = db.Column(db.Integer, nullable=False)  # 1-5 scale
    years_experience = db.Column(db.Integer, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    priority = db.Column(db.String(20), nullable=False)
    estimated_duration_months = db.Column(db.Integer, nullable=False)
    budget = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ProjectSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    skill_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)
    importance_level = db.Column(db.Integer, nullable=False)  # 1-5 scale
    required_proficiency = db.Column(db.Integer, nullable=False)  # 1-5 scale

class DevelopmentPath(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    estimated_duration_months = db.Column(db.Integer, nullable=False)
    target_roles = db.Column(db.String(500))

class MatchingResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    match_score = db.Column(db.Float, nullable=False)
    skill_gap_analysis = db.Column(db.Text)
    recommended_training = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# AI Matching Engine Class
class TalentMatchingEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def calculate_skill_match_score(self, employee_id, project_id):
        """Calculate skill-based matching score between employee and project"""
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

            # Create skill vectors
            employee_skill_dict = {skill.skill_name: emp_skill.proficiency_level 
                                 for emp_skill, skill in employee_skills}

            total_score = 0.0
            max_possible_score = 0.0

            for proj_skill, skill in project_skills:
                importance_weight = proj_skill.importance_level / 5.0
                required_proficiency = proj_skill.required_proficiency

                employee_proficiency = employee_skill_dict.get(skill.skill_name, 0)

                # Calculate skill score with proficiency matching
                if employee_proficiency >= required_proficiency:
                    skill_score = 1.0
                elif employee_proficiency > 0:
                    skill_score = employee_proficiency / required_proficiency * 0.7
                else:
                    skill_score = 0.0

                weighted_score = skill_score * importance_weight
                total_score += weighted_score
                max_possible_score += importance_weight

            # Normalize score
            final_score = (total_score / max_possible_score) if max_possible_score > 0 else 0.0
            return min(final_score, 1.0)

        except Exception as e:
            print(f"Error calculating skill match score: {e}")
            return 0.0

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

    def semantic_text_similarity(self, text1, text2):
        """Calculate semantic similarity between texts using TF-IDF"""
        try:
            if not text1 or not text2:
                return 0.0

            texts = [text1.lower(), text2.lower()]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return similarity_matrix[0][0]

        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0

    def calculate_career_alignment_score(self, employee_id, project_id):
        """Calculate how well the project aligns with employee career goals"""
        try:
            employee = Employee.query.get(employee_id)
            project = Project.query.get(project_id)

            if not employee or not project or not employee.career_aspirations:
                return 0.5  # Neutral score if no career info

            # Semantic similarity between career aspirations and project description
            career_score = self.semantic_text_similarity(
                employee.career_aspirations,
                project.description
            )

            # Role alignment bonus
            role_keywords = employee.role.lower().split()
            project_keywords = project.description.lower().split()
            role_overlap = len(set(role_keywords) & set(project_keywords)) / len(role_keywords)

            final_score = (career_score * 0.7) + (role_overlap * 0.3)
            return min(final_score, 1.0)

        except Exception as e:
            print(f"Error calculating career alignment: {e}")
            return 0.5

    def generate_comprehensive_match_score(self, employee_id, project_id):
        """Generate comprehensive matching score using multiple factors"""
        try:
            # Weight factors for different components
            weights = {
                'skill_match': 0.45,
                'experience': 0.25,
                'career_alignment': 0.20,
                'availability': 0.10
            }

            # Calculate individual scores
            skill_score = self.calculate_skill_match_score(employee_id, project_id)
            experience_score = self.calculate_experience_compatibility(employee_id, project_id)
            career_score = self.calculate_career_alignment_score(employee_id, project_id)
            availability_score = 0.8  # Placeholder - could be enhanced with calendar data

            # Calculate weighted final score
            final_score = (
                skill_score * weights['skill_match'] +
                experience_score * weights['experience'] +
                career_score * weights['career_alignment'] +
                availability_score * weights['availability']
            )

            return {
                'overall_score': final_score,
                'skill_match': skill_score,
                'experience_compatibility': experience_score,
                'career_alignment': career_score,
                'availability': availability_score,
                'components': {
                    'skill_details': self.get_skill_gap_analysis(employee_id, project_id),
                    'recommendation': self.generate_recommendation(final_score, skill_score)
                }
            }

        except Exception as e:
            print(f"Error generating comprehensive match score: {e}")
            return {'overall_score': 0.0, 'error': str(e)}

    def get_skill_gap_analysis(self, employee_id, project_id):
        """Analyze skill gaps and provide detailed breakdown"""
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
        """Generate recommendation based on scores"""
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

            development_paths = DevelopmentPath.query.all()
            recommendations = []

            for path in development_paths:
                # Calculate alignment score based on career aspirations and current role
                alignment_score = 0.0

                if employee.career_aspirations:
                    alignment_score += self.semantic_text_similarity(
                        employee.career_aspirations, path.target_roles
                    ) * 0.6

                # Check if current role aligns with path
                role_alignment = self.semantic_text_similarity(
                    employee.role, path.description
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

# Initialize matching engine
matching_engine = TalentMatchingEngine()

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

def initialize_database():
    """Initialize database with sample data"""
    with app.app_context():
        # Create all tables
        db.create_all()

        # Check if data already exists
        if Employee.query.count() > 0:
            print("Database already contains data. Skipping initialization.")
            return

        try:
            # Load and insert employees
            employees_df = pd.read_csv('employees_data.csv')
            for _, row in employees_df.iterrows():
                employee = Employee(
                    name=row['name'],
                    department=row['department'],
                    role=row['role'],
                    experience_years=row['experience_years'],
                    current_salary=row['current_salary'],
                    location=row['location'],
                    career_aspirations=row['career_aspirations']
                )
                db.session.add(employee)

            # Load and insert skills
            skills_df = pd.read_csv('skills_data.csv')
            for _, row in skills_df.iterrows():
                skill = Skill(
                    skill_name=row['skill_name'],
                    category=row['category']
                )
                db.session.add(skill)

            # Load and insert projects
            projects_df = pd.read_csv('projects_data.csv')
            for _, row in projects_df.iterrows():
                project = Project(
                    project_name=row['project_name'],
                    description=row['description'],
                    priority=row['priority'],
                    estimated_duration_months=row['estimated_duration_months'],
                    budget=row['budget'],
                    status=row['status']
                )
                db.session.add(project)

            # Load and insert development paths
            dev_paths_df = pd.read_csv('development_paths_data.csv')
            for _, row in dev_paths_df.iterrows():
                dev_path = DevelopmentPath(
                    path_name=row['path_name'],
                    description=row['description'],
                    estimated_duration_months=row['estimated_duration_months'],
                    target_roles=row['target_roles']
                )
                db.session.add(dev_path)

            # Commit all the basic data first
            db.session.commit()

            # Load and insert employee skills
            emp_skills_df = pd.read_csv('employee_skills_data.csv')
            for _, row in emp_skills_df.iterrows():
                emp_skill = EmployeeSkill(
                    employee_id=row['employee_id'],
                    skill_id=row['skill_id'],
                    proficiency_level=row['proficiency_level'],
                    years_experience=row['years_experience']
                )
                db.session.add(emp_skill)

            # Load and insert project skills
            proj_skills_df = pd.read_csv('project_skills_data.csv')
            for _, row in proj_skills_df.iterrows():
                proj_skill = ProjectSkill(
                    project_id=row['project_id'],
                    skill_id=row['skill_id'],
                    importance_level=row['importance_level'],
                    required_proficiency=row['required_proficiency']
                )
                db.session.add(proj_skill)

            db.session.commit()
            print("Database initialized successfully with sample data!")

        except Exception as e:
            db.session.rollback()
            print(f"Error initializing database: {e}")

if __name__ == '__main__':
    initialize_database()
    app.run(debug=True, host='0.0.0.0', port=5000)
