import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db, Employee, Project, Skill, EmployeeSkill, ProjectSkill, TalentMatchingEngine
import pandas as pd

class TestTalentMatching(unittest.TestCase):
    """Test cases for the talent matching system"""

    def setUp(self):
        """Set up test environment"""
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()
            self.create_test_data()
            self.matching_engine = TalentMatchingEngine()

    def tearDown(self):
        """Clean up test environment"""
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def create_test_data(self):
        """Create test data"""
        # Create test employee
        employee = Employee(
            name="Test Employee",
            department="Engineering",
            role="Senior Developer",
            experience_years=5,
            current_salary=90000,
            location="New York",
            career_aspirations="Tech Lead"
        )
        db.session.add(employee)

        # Create test skills
        python_skill = Skill(skill_name="Python", category="Technical")
        ml_skill = Skill(skill_name="Machine Learning", category="Technical")
        db.session.add(python_skill)
        db.session.add(ml_skill)

        # Create test project
        project = Project(
            project_name="Test Project",
            description="A test project using Python and Machine Learning",
            priority="High",
            estimated_duration_months=6,
            budget=100000,
            status="Planning"
        )
        db.session.add(project)

        db.session.commit()

        # Create employee skills
        emp_skill1 = EmployeeSkill(
            employee_id=employee.id,
            skill_id=python_skill.id,
            proficiency_level=4,
            years_experience=3
        )
        emp_skill2 = EmployeeSkill(
            employee_id=employee.id,
            skill_id=ml_skill.id,
            proficiency_level=3,
            years_experience=2
        )
        db.session.add(emp_skill1)
        db.session.add(emp_skill2)

        # Create project skills
        proj_skill1 = ProjectSkill(
            project_id=project.id,
            skill_id=python_skill.id,
            importance_level=5,
            required_proficiency=4
        )
        proj_skill2 = ProjectSkill(
            project_id=project.id,
            skill_id=ml_skill.id,
            importance_level=4,
            required_proficiency=3
        )
        db.session.add(proj_skill1)
        db.session.add(proj_skill2)

        db.session.commit()

    def test_skill_match_calculation(self):
        """Test skill matching calculation"""
        with self.app.app_context():
            score = self.matching_engine.calculate_skill_match_score(1, 1)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_experience_compatibility(self):
        """Test experience compatibility calculation"""
        with self.app.app_context():
            score = self.matching_engine.calculate_experience_compatibility(1, 1)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_comprehensive_match_score(self):
        """Test comprehensive matching score"""
        with self.app.app_context():
            result = self.matching_engine.generate_comprehensive_match_score(1, 1)
            self.assertIn('overall_score', result)
            self.assertIn('skill_match', result)
            self.assertIn('experience_compatibility', result)
            self.assertIn('career_alignment', result)
            self.assertIsInstance(result['overall_score'], float)

    def test_skill_gap_analysis(self):
        """Test skill gap analysis"""
        with self.app.app_context():
            analysis = self.matching_engine.get_skill_gap_analysis(1, 1)
            self.assertIn('strengths', analysis)
            self.assertIn('gaps', analysis)
            self.assertIsInstance(analysis['strengths'], list)
            self.assertIsInstance(analysis['gaps'], list)

    def test_find_best_matches(self):
        """Test finding best matches for project"""
        with self.app.app_context():
            matches = self.matching_engine.find_best_matches_for_project(1, limit=5)
            self.assertIsInstance(matches, list)
            self.assertLessEqual(len(matches), 5)

    def test_api_endpoints(self):
        """Test API endpoints"""
        with self.app.app_context():
            # Test match score API
            response = self.client.get('/api/match-score?employee_id=1&project_id=1')
            self.assertEqual(response.status_code, 200)

            # Test skill gap API
            response = self.client.get('/api/skill-gap/1/1')
            self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
