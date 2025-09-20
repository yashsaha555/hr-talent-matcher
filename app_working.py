#!/usr/bin/env python3
"""
Working version of SkillBridge HR Talent Matching System
Simplified to work with available dependencies
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import sqlite3
import json
import re

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hr_talent_matching.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'

db = SQLAlchemy(app)

# Basic Database Models
class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    experience_years = db.Column(db.Integer, default=0)
    location = db.Column(db.String(100))
    availability_status = db.Column(db.String(50), default='Available')
    career_aspirations = db.Column(db.Text)
    personality_type = db.Column(db.String(50))
    work_preferences = db.Column(db.Text)
    certifications = db.Column(db.Text)
    languages = db.Column(db.Text)
    resume_text = db.Column(db.Text)
    linkedin_profile = db.Column(db.String(200))
    github_profile = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Skill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    skill_name = db.Column(db.String(100), nullable=False, unique=True)
    category = db.Column(db.String(50))
    subcategory = db.Column(db.String(50))
    skill_type = db.Column(db.String(50))  # Technical, Soft, Domain
    market_demand = db.Column(db.Float, default=0.5)
    future_relevance = db.Column(db.Float, default=0.5)
    description = db.Column(db.Text)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(db.String(50), default='Planning')
    priority = db.Column(db.String(50), default='Medium')
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    estimated_duration_months = db.Column(db.Integer)
    budget = db.Column(db.Float)
    complexity_score = db.Column(db.Float, default=0.5)
    success_probability = db.Column(db.Float, default=0.5)
    team_size_required = db.Column(db.Integer, default=1)
    remote_friendly = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EmployeeSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    skill_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)
    proficiency_level = db.Column(db.Integer, default=1)  # 1-5 scale
    years_of_experience = db.Column(db.Float, default=0)
    last_used = db.Column(db.Date)
    certified = db.Column(db.Boolean, default=False)

class ProjectSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    skill_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)
    required_level = db.Column(db.Integer, default=1)  # 1-5 scale
    importance = db.Column(db.String(20), default='Medium')  # Low, Medium, High, Critical

# Basic Matching Engine
class BasicTalentMatchingEngine:
    def __init__(self):
        self.models_trained = False

    def calculate_skill_match_score(self, employee_id, project_id):
        """Basic skill matching calculation"""
        try:
            # Get employee skills
            employee_skills = db.session.query(EmployeeSkill, Skill).join(
                Skill, EmployeeSkill.skill_id == Skill.id
            ).filter(EmployeeSkill.employee_id == employee_id).all()

            # Get project requirements
            project_skills = db.session.query(ProjectSkill, Skill).join(
                Skill, ProjectSkill.skill_id == Skill.id
            ).filter(ProjectSkill.project_id == project_id).all()

            if not project_skills:
                return 0.0

            total_score = 0
            total_weight = 0

            for proj_skill, skill in project_skills:
                # Find matching employee skill
                emp_skill = next(
                    (es for es, s in employee_skills if s.id == skill.id), 
                    None
                )

                if emp_skill:
                    # Calculate proficiency match (0-1 scale)
                    proficiency_match = min(emp_skill.proficiency_level / proj_skill.required_level, 1.0)
                else:
                    proficiency_match = 0.0

                # Weight by importance
                weight = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}.get(proj_skill.importance, 2)
                total_score += proficiency_match * weight
                total_weight += weight

            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            print(f"Error calculating skill match: {e}")
            return 0.0

    def get_employee_recommendations(self, project_id, limit=10):
        """Get recommended employees for a project"""
        try:
            employees = Employee.query.filter_by(availability_status='Available').all()
            recommendations = []

            for employee in employees:
                skill_match = self.calculate_skill_match_score(employee.id, project_id)
                
                recommendations.append({
                    'employee': employee,
                    'skill_match_percentage': round(skill_match * 100, 2),
                    'overall_score': skill_match,
                    'recommendation': 'Excellent match' if skill_match > 0.8 else 
                                   'Good match' if skill_match > 0.6 else 
                                   'Fair match' if skill_match > 0.4 else 'Poor match'
                })

            # Sort by overall score
            recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
            return recommendations[:limit]

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

# Initialize matching engine
matching_engine = BasicTalentMatchingEngine()

# Routes
@app.route('/')
def index():
    """Main dashboard"""
    try:
        employee_count = Employee.query.count()
        project_count = Project.query.count()
        skill_count = Skill.query.count()
        active_projects = Project.query.filter_by(status='Active').count()

        stats = {
            'employees': employee_count,
            'projects': project_count,
            'skills': skill_count,
            'active_projects': active_projects
        }

        return render_template('dashboard.html', stats=stats)
    except Exception as e:
        return f"Dashboard Error: {str(e)}"

@app.route('/employees')
def employees():
    """Employees page"""
    try:
        employees = Employee.query.all()
        return render_template('employees.html', employees=employees)
    except Exception as e:
        return f"Employees Error: {str(e)}"

@app.route('/projects')
def projects():
    """Projects page"""
    try:
        projects = Project.query.all()
        return render_template('projects.html', projects=projects)
    except Exception as e:
        return f"Projects Error: {str(e)}"

@app.route('/skills')
def skills():
    """Skills page"""
    try:
        skills = Skill.query.all()
        return render_template('skills.html', skills=skills)
    except Exception as e:
        return f"Skills Error: {str(e)}"

# AI Feature Pages (Basic versions)
@app.route('/resume-analysis')
def resume_analysis():
    """Resume analysis page"""
    try:
        employees = Employee.query.all()
        return render_template('resume_analysis.html', employees=employees)
    except Exception as e:
        return f"Resume Analysis Error: {str(e)}"

@app.route('/performance-prediction')
def performance_prediction():
    """Performance prediction page"""
    try:
        employees = Employee.query.all()
        projects = Project.query.all()
        return render_template('performance_prediction.html', employees=employees, projects=projects)
    except Exception as e:
        return f"Performance Prediction Error: {str(e)}"

@app.route('/team-optimization')
def team_optimization():
    """Team optimization page"""
    try:
        projects = Project.query.all()
        return render_template('team_optimization.html', projects=projects)
    except Exception as e:
        return f"Team Optimization Error: {str(e)}"

@app.route('/career-planning')
def career_planning():
    """Career planning page"""
    try:
        employees = Employee.query.all()
        career_paths = []  # Mock empty list for now
        return render_template('career_planning.html', employees=employees, career_paths=career_paths)
    except Exception as e:
        return f"Career Planning Error: {str(e)}"

@app.route('/training-recommendations')
def training_recommendations():
    """Training recommendations page"""
    try:
        employees = Employee.query.all()
        courses = []  # Mock empty list for now
        return render_template('training_recommendations.html', employees=employees, courses=courses)
    except Exception as e:
        return f"Training Recommendations Error: {str(e)}"

# API Endpoints
@app.route('/api/employee-recommendations/<int:project_id>')
def api_employee_recommendations(project_id):
    """API endpoint for employee recommendations"""
    try:
        recommendations = matching_engine.get_employee_recommendations(project_id)
        
        result = []
        for rec in recommendations:
            employee = rec['employee']
            result.append({
                'id': employee.id,
                'name': employee.name,
                'role': employee.role,
                'department': employee.department,
                'experience_years': employee.experience_years,
                'skill_match_percentage': rec['skill_match_percentage'],
                'overall_score': rec['overall_score'],
                'recommendation': rec['recommendation']
            })
        
        return jsonify({'recommendations': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test')
def api_test():
    """Test API endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'SkillBridge API is working!',
        'employees': Employee.query.count(),
        'projects': Project.query.count(),
        'version': 'Basic Working Version'
    })

def initialize_sample_data():
    """Initialize database with sample data"""
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Check if data already exists
        if Employee.query.count() == 0:
            print("Initializing sample data...")
            
            # Add sample skills
            skills_data = [
                {'skill_name': 'Python', 'category': 'Programming', 'skill_type': 'Technical'},
                {'skill_name': 'JavaScript', 'category': 'Programming', 'skill_type': 'Technical'},
                {'skill_name': 'Machine Learning', 'category': 'AI/ML', 'skill_type': 'Technical'},
                {'skill_name': 'Project Management', 'category': 'Management', 'skill_type': 'Soft'},
                {'skill_name': 'Data Analysis', 'category': 'Analytics', 'skill_type': 'Technical'},
                {'skill_name': 'UI/UX Design', 'category': 'Design', 'skill_type': 'Technical'},
                {'skill_name': 'Communication', 'category': 'Soft Skills', 'skill_type': 'Soft'},
                {'skill_name': 'Leadership', 'category': 'Management', 'skill_type': 'Soft'}
            ]
            
            for skill_data in skills_data:
                skill = Skill(**skill_data)
                db.session.add(skill)
            
            db.session.flush()  # Get IDs
            
            # Add sample employees
            employees_data = [
                {
                    'name': 'Alice Johnson',
                    'email': 'alice.johnson@company.com',
                    'role': 'Software Developer',
                    'department': 'Engineering',
                    'experience_years': 5,
                    'location': 'New York',
                    'career_aspirations': 'Senior Developer, Tech Lead'
                },
                {
                    'name': 'Bob Smith',
                    'email': 'bob.smith@company.com',
                    'role': 'Data Scientist',
                    'department': 'Analytics',
                    'experience_years': 3,
                    'location': 'San Francisco',
                    'career_aspirations': 'ML Engineer, Data Architect'
                },
                {
                    'name': 'Carol Davis',
                    'email': 'carol.davis@company.com',
                    'role': 'Project Manager',
                    'department': 'Operations',
                    'experience_years': 7,
                    'location': 'Chicago',
                    'career_aspirations': 'Program Manager, Director'
                },
                {
                    'name': 'David Wilson',
                    'email': 'david.wilson@company.com',
                    'role': 'UX Designer',
                    'department': 'Design',
                    'experience_years': 4,
                    'location': 'Austin',
                    'career_aspirations': 'Senior Designer, Design Lead'
                }
            ]
            
            for emp_data in employees_data:
                employee = Employee(**emp_data)
                db.session.add(employee)
            
            db.session.flush()
            
            # Add sample projects
            projects_data = [
                {
                    'project_name': 'AI Customer Service Bot',
                    'description': 'Develop an AI-powered customer service chatbot using machine learning',
                    'status': 'Active',
                    'priority': 'High',
                    'estimated_duration_months': 6,
                    'budget': 150000,
                    'team_size_required': 3
                },
                {
                    'project_name': 'Data Analytics Dashboard',
                    'description': 'Create a comprehensive analytics dashboard for business intelligence',
                    'status': 'Planning',
                    'priority': 'Medium',
                    'estimated_duration_months': 4,
                    'budget': 80000,
                    'team_size_required': 2
                },
                {
                    'project_name': 'Mobile App Redesign',
                    'description': 'Redesign the company mobile application with modern UX principles',
                    'status': 'Active',
                    'priority': 'High',
                    'estimated_duration_months': 8,
                    'budget': 200000,
                    'team_size_required': 4
                }
            ]
            
            for proj_data in projects_data:
                project = Project(**proj_data)
                db.session.add(project)
            
            db.session.commit()
            print("Sample data initialized successfully!")

if __name__ == '__main__':
    initialize_sample_data()
    print("Starting SkillBridge Application...")
    print("Dashboard: http://127.0.0.1:5000")
    print("API Test: http://127.0.0.1:5000/api/test")
    app.run(debug=True, host='127.0.0.1', port=5000)
