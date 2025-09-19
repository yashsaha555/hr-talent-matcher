import json
import pandas as pd
from flask import jsonify
from datetime import datetime

def create_api_response(data, success=True, message="", status_code=200):
    """Create standardized API response"""
    response = {
        'success': success,
        'message': message,
        'data': data,
        'timestamp': datetime.utcnow().isoformat()
    }
    return jsonify(response), status_code

def handle_api_error(error, status_code=500):
    """Handle API errors consistently"""
    return create_api_response(
        data=None,
        success=False,
        message=str(error),
        status_code=status_code
    )

def validate_employee_data(data):
    """Validate employee data for API requests"""
    required_fields = ['name', 'department', 'role', 'experience_years']
    errors = []

    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")

    if 'experience_years' in data:
        try:
            years = int(data['experience_years'])
            if years < 0 or years > 50:
                errors.append("Experience years must be between 0 and 50")
        except (ValueError, TypeError):
            errors.append("Experience years must be a valid number")

    return errors

def validate_project_data(data):
    """Validate project data for API requests"""
    required_fields = ['project_name', 'description', 'priority', 'estimated_duration_months']
    errors = []

    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")

    if 'priority' in data and data['priority'] not in ['High', 'Medium', 'Low']:
        errors.append("Priority must be High, Medium, or Low")

    if 'estimated_duration_months' in data:
        try:
            duration = int(data['estimated_duration_months'])
            if duration < 1 or duration > 60:
                errors.append("Duration must be between 1 and 60 months")
        except (ValueError, TypeError):
            errors.append("Duration must be a valid number")

    return errors

def format_skill_data(employee_skills, project_skills):
    """Format skill data for API responses"""
    return {
        'employee_skills': [
            {
                'skill_name': skill.skill_name,
                'proficiency': emp_skill.proficiency_level,
                'years_experience': emp_skill.years_experience,
                'category': skill.category
            }
            for emp_skill, skill in employee_skills
        ],
        'project_requirements': [
            {
                'skill_name': skill.skill_name,
                'required_proficiency': proj_skill.required_proficiency,
                'importance': proj_skill.importance_level,
                'category': skill.category
            }
            for proj_skill, skill in project_skills
        ]
    }
