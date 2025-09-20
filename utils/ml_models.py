"""
Machine Learning Models for Performance Prediction and Advanced Matching
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PerformancePredictionModel:
    """ML model to predict employee performance on projects"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def prepare_features(self, employee_data, project_data, historical_data=None):
        """Prepare features for performance prediction"""
        features = []
        
        # Employee features
        features.extend([
            employee_data.get('experience_years', 0),
            employee_data.get('current_salary', 0) / 100000,  # Normalize salary
            len(employee_data.get('skills', [])),
            employee_data.get('avg_skill_level', 0),
            employee_data.get('education_score', 0),
            employee_data.get('certification_count', 0)
        ])
        
        # Project features
        features.extend([
            project_data.get('complexity_score', 0),
            project_data.get('estimated_duration_months', 0),
            project_data.get('budget', 0) / 1000000,  # Normalize budget
            project_data.get('team_size_required', 1),
            1 if project_data.get('priority') == 'High' else 0,
            1 if project_data.get('remote_friendly', True) else 0
        ])
        
        # Skill match features
        features.extend([
            employee_data.get('skill_match_score', 0),
            employee_data.get('experience_match_score', 0),
            employee_data.get('career_alignment_score', 0)
        ])
        
        # Historical performance features (if available)
        if historical_data:
            features.extend([
                historical_data.get('avg_past_performance', 0),
                historical_data.get('project_completion_rate', 0),
                historical_data.get('similar_project_performance', 0)
            ])
        else:
            features.extend([0, 0, 0])
        
        self.feature_names = [
            'experience_years', 'salary_normalized', 'skill_count', 'avg_skill_level',
            'education_score', 'certification_count', 'project_complexity',
            'duration_months', 'budget_normalized', 'team_size', 'high_priority',
            'remote_friendly', 'skill_match_score', 'experience_match_score',
            'career_alignment_score', 'avg_past_performance', 'completion_rate',
            'similar_project_performance'
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data):
        """Train the performance prediction model"""
        if len(training_data) < 10:
            print("Warning: Insufficient training data. Using default model.")
            return False
        
        X = []
        y = []
        
        for record in training_data:
            features = self.prepare_features(
                record['employee_data'],
                record['project_data'],
                record.get('historical_data')
            )
            X.append(features.flatten())
            y.append(record['performance_rating'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training score
        train_score = self.model.score(X_scaled, y)
        print(f"Performance prediction model trained with R² score: {train_score:.3f}")
        
        return True
    
    def predict_performance(self, employee_data, project_data, historical_data=None):
        """Predict employee performance on a project"""
        if not self.is_trained:
            # Return default prediction if model not trained
            return {
                'predicted_performance': 3.5,
                'confidence': 0.5,
                'factors': {'default': 'Model not trained'}
            }
        
        features = self.prepare_features(employee_data, project_data, historical_data)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence based on feature importance
        feature_importance = self.model.feature_importances_
        confidence = min(0.9, max(0.1, np.mean(feature_importance)))
        
        # Get top contributing factors
        feature_contributions = dict(zip(self.feature_names, feature_importance))
        top_factors = dict(sorted(feature_contributions.items(), 
                                key=lambda x: x[1], reverse=True)[:5])
        
        return {
            'predicted_performance': max(1.0, min(5.0, prediction)),
            'confidence': confidence,
            'factors': top_factors
        }

class SuccessProbabilityModel:
    """ML model to predict project success probability"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def prepare_features(self, project_data, team_data, historical_data=None):
        """Prepare features for success probability prediction"""
        features = []
        
        # Project features
        features.extend([
            project_data.get('complexity_score', 0),
            project_data.get('estimated_duration_months', 0),
            project_data.get('budget', 0) / 1000000,
            1 if project_data.get('priority') == 'High' else 0,
            1 if project_data.get('remote_friendly', True) else 0
        ])
        
        # Team features
        team_size = len(team_data.get('members', []))
        avg_experience = np.mean([m.get('experience_years', 0) for m in team_data.get('members', [])]) if team_size > 0 else 0
        avg_skill_match = np.mean([m.get('skill_match_score', 0) for m in team_data.get('members', [])]) if team_size > 0 else 0
        skill_diversity = len(set([skill for m in team_data.get('members', []) for skill in m.get('skills', [])])) if team_size > 0 else 0
        
        features.extend([
            team_size,
            avg_experience,
            avg_skill_match,
            skill_diversity,
            team_data.get('team_synergy_score', 0)
        ])
        
        # Historical features
        if historical_data:
            features.extend([
                historical_data.get('similar_project_success_rate', 0),
                historical_data.get('team_past_success_rate', 0),
                historical_data.get('client_satisfaction_history', 0)
            ])
        else:
            features.extend([0, 0, 0])
        
        self.feature_names = [
            'complexity_score', 'duration_months', 'budget_normalized', 'high_priority',
            'remote_friendly', 'team_size', 'avg_experience', 'avg_skill_match',
            'skill_diversity', 'team_synergy_score', 'similar_project_success_rate',
            'team_past_success_rate', 'client_satisfaction_history'
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data):
        """Train the success probability model"""
        if len(training_data) < 10:
            print("Warning: Insufficient training data for success prediction.")
            return False
        
        X = []
        y = []
        
        for record in training_data:
            features = self.prepare_features(
                record['project_data'],
                record['team_data'],
                record.get('historical_data')
            )
            X.append(features.flatten())
            y.append(1 if record['project_success'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training accuracy
        train_accuracy = self.model.score(X_scaled, y)
        print(f"Success probability model trained with accuracy: {train_accuracy:.3f}")
        
        return True
    
    def predict_success_probability(self, project_data, team_data, historical_data=None):
        """Predict project success probability"""
        if not self.is_trained:
            # Return default prediction
            return {
                'success_probability': 0.7,
                'confidence': 0.5,
                'risk_factors': ['Model not trained']
            }
        
        features = self.prepare_features(project_data, team_data, historical_data)
        features_scaled = self.scaler.transform(features)
        
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        # Identify risk factors
        feature_importance = self.model.feature_importances_
        feature_values = features.flatten()
        
        risk_factors = []
        for i, (name, importance, value) in enumerate(zip(self.feature_names, feature_importance, feature_values)):
            if importance > 0.1 and value < 0.5:  # High importance but low value
                risk_factors.append(name)
        
        confidence = min(0.9, max(0.1, np.mean(feature_importance)))
        
        return {
            'success_probability': probability,
            'confidence': confidence,
            'risk_factors': risk_factors[:5]  # Top 5 risk factors
        }

class TeamOptimizationModel:
    """ML model for optimal team composition"""
    
    def __init__(self):
        self.skill_importance_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.synergy_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_trained = False
    
    def calculate_team_synergy(self, team_members):
        """Calculate team synergy score based on member characteristics"""
        if len(team_members) < 2:
            return 0.5
        
        synergy_factors = []
        
        # Skill complementarity
        all_skills = set()
        skill_overlaps = 0
        for member in team_members:
            member_skills = set(member.get('skills', []))
            skill_overlaps += len(all_skills.intersection(member_skills))
            all_skills.update(member_skills)
        
        skill_diversity = len(all_skills) / (len(team_members) * 10)  # Normalize
        skill_overlap_ratio = skill_overlaps / len(all_skills) if all_skills else 0
        
        synergy_factors.extend([skill_diversity, skill_overlap_ratio])
        
        # Experience diversity
        experiences = [m.get('experience_years', 0) for m in team_members]
        exp_std = np.std(experiences) / np.mean(experiences) if np.mean(experiences) > 0 else 0
        synergy_factors.append(min(1.0, exp_std))
        
        # Department diversity
        departments = set([m.get('department', '') for m in team_members])
        dept_diversity = len(departments) / len(team_members)
        synergy_factors.append(dept_diversity)
        
        # Location compatibility (for remote work)
        locations = [m.get('location', '') for m in team_members]
        same_location_ratio = max([locations.count(loc) for loc in set(locations)]) / len(locations)
        synergy_factors.append(1 - same_location_ratio)  # Higher diversity = better synergy
        
        return np.mean(synergy_factors)
    
    def optimize_team_for_project(self, project_requirements, available_employees, max_team_size=5):
        """Find optimal team composition for a project"""
        from scipy.optimize import linear_sum_assignment
        
        # Calculate match scores for all employees
        employee_scores = []
        for emp in available_employees:
            score = self.calculate_employee_project_fit(emp, project_requirements)
            employee_scores.append((emp, score))
        
        # Sort by individual fit
        employee_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy team building with synergy optimization
        best_team = []
        remaining_employees = [emp for emp, _ in employee_scores]
        
        # Start with the best individual performer
        if remaining_employees:
            best_team.append(remaining_employees.pop(0))
        
        # Add team members that maximize overall team performance
        while len(best_team) < max_team_size and remaining_employees:
            best_addition = None
            best_team_score = 0
            
            for candidate in remaining_employees[:10]:  # Consider top 10 candidates
                test_team = best_team + [candidate]
                team_score = self.calculate_team_score(test_team, project_requirements)
                
                if team_score > best_team_score:
                    best_team_score = team_score
                    best_addition = candidate
            
            if best_addition:
                best_team.append(best_addition)
                remaining_employees.remove(best_addition)
            else:
                break
        
        # Calculate final team metrics
        team_synergy = self.calculate_team_synergy(best_team)
        total_cost = sum([emp.get('current_salary', 0) for emp in best_team])
        skill_coverage = self.calculate_skill_coverage(best_team, project_requirements)
        
        return {
            'recommended_team': best_team,
            'team_size': len(best_team),
            'team_synergy_score': team_synergy,
            'total_cost': total_cost,
            'skill_coverage': skill_coverage,
            'estimated_success_rate': min(0.95, team_synergy * skill_coverage)
        }
    
    def calculate_employee_project_fit(self, employee, project_requirements):
        """Calculate how well an employee fits a project"""
        fit_score = 0.0
        
        # Skill match
        emp_skills = set(employee.get('skills', []))
        req_skills = set(project_requirements.get('required_skills', []))
        
        if req_skills:
            skill_match = len(emp_skills.intersection(req_skills)) / len(req_skills)
            fit_score += skill_match * 0.4
        
        # Experience match
        req_experience = project_requirements.get('min_experience', 0)
        emp_experience = employee.get('experience_years', 0)
        
        if req_experience > 0:
            exp_match = min(1.0, emp_experience / req_experience)
            fit_score += exp_match * 0.3
        
        # Availability
        if employee.get('availability_status') == 'Available':
            fit_score += 0.2
        
        # Location match (for non-remote projects)
        if not project_requirements.get('remote_friendly', True):
            if employee.get('location') == project_requirements.get('location'):
                fit_score += 0.1
        
        return fit_score
    
    def calculate_team_score(self, team, project_requirements):
        """Calculate overall team score for a project"""
        individual_scores = [self.calculate_employee_project_fit(emp, project_requirements) 
                           for emp in team]
        avg_individual_score = np.mean(individual_scores)
        
        team_synergy = self.calculate_team_synergy(team)
        skill_coverage = self.calculate_skill_coverage(team, project_requirements)
        
        # Weighted combination
        team_score = (avg_individual_score * 0.4 + 
                     team_synergy * 0.3 + 
                     skill_coverage * 0.3)
        
        return team_score
    
    def calculate_skill_coverage(self, team, project_requirements):
        """Calculate how well the team covers required skills"""
        req_skills = set(project_requirements.get('required_skills', []))
        if not req_skills:
            return 1.0
        
        covered_skills = set()
        for member in team:
            covered_skills.update(member.get('skills', []))
        
        coverage = len(req_skills.intersection(covered_skills)) / len(req_skills)
        return coverage

class CareerPathPredictor:
    """ML model for career path recommendations and progression prediction"""
    
    def __init__(self):
        self.progression_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.timeline_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def prepare_career_features(self, employee_data):
        """Prepare features for career progression prediction"""
        features = []
        
        # Current role and experience
        features.extend([
            employee_data.get('experience_years', 0),
            len(employee_data.get('skills', [])),
            employee_data.get('avg_performance_rating', 0),
            employee_data.get('education_score', 0),
            employee_data.get('certification_count', 0)
        ])
        
        # Recent performance trends
        features.extend([
            employee_data.get('performance_trend', 0),  # Positive/negative trend
            employee_data.get('skill_acquisition_rate', 0),
            employee_data.get('project_success_rate', 0),
            employee_data.get('leadership_score', 0)
        ])
        
        # Market factors
        features.extend([
            employee_data.get('role_market_demand', 0),
            employee_data.get('skill_market_relevance', 0)
        ])
        
        return np.array(features).reshape(1, -1)
    
    def predict_career_progression(self, employee_data, target_roles):
        """Predict career progression possibilities"""
        if not self.is_trained:
            # Return default predictions
            return {
                'recommended_paths': target_roles[:3],
                'success_probabilities': [0.7, 0.6, 0.5],
                'estimated_timelines': [24, 36, 48],  # months
                'required_skills': []
            }
        
        features = self.prepare_career_features(employee_data)
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        for role in target_roles:
            # Predict success probability for this career path
            prob = self.progression_model.predict_proba(features_scaled)[0][1]
            
            # Predict timeline
            timeline = self.timeline_model.predict(features_scaled)[0]
            
            predictions.append({
                'target_role': role,
                'success_probability': prob,
                'estimated_timeline_months': max(6, int(timeline)),
                'confidence': 0.7  # Default confidence
            })
        
        # Sort by success probability
        predictions.sort(key=lambda x: x['success_probability'], reverse=True)
        
        return {
            'career_predictions': predictions[:5],
            'current_readiness_score': np.mean([p['success_probability'] for p in predictions])
        }
