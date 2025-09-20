# AI-Powered HR Talent Matching System

## Overview

This is a comprehensive AI-powered talent matching and development system built for the hackathon. The system analyzes employee skills, preferences, and career aspirations to recommend optimal project assignments and development opportunities using advanced machine learning algorithms.

## Features

### 🧠 AI-Powered Matching Engine
- **Multi-factor Analysis**: Combines skill matching, experience compatibility, career alignment, and availability
- **Weighted Scoring**: Configurable weights for different matching criteria
- **Semantic Analysis**: Uses TF-IDF vectorization and cosine similarity for text analysis
- **Real-time Calculations**: Dynamic matching scores with detailed breakdowns.

### 📊 Skill Gap Analysis
- **Detailed Gap Identification**: Identifies specific skill gaps and proficiency levels
- **Strength Assessment**: Highlights employee strengths and suitable areas
- **Training Recommendations**: Suggests development paths based on gap analysis
- **Visual Reporting**: Interactive charts and progress indicators

### 🚀 Career Development
- **Personalized Paths**: AI-recommended development tracks aligned with career goals
- **Duration Estimates**: Realistic timeframes for skill development
- **Target Role Mapping**: Clear progression paths to desired positions
- **Learning Resource Integration**: Links to training materials and courses

### 📈 Project Matching
- **Optimal Team Formation**: AI-suggested team compositions for projects
- **Priority-based Matching**: Considers project priority and urgency
- **Budget Alignment**: Matches based on project budget and employee salary expectations
- **Timeline Compatibility**: Ensures availability and duration alignment

## Technical Architecture

### Backend Components
- **Flask Web Framework**: RESTful API and web interface
- **SQLAlchemy ORM**: Database management and relationships
- **Scikit-learn**: Machine learning algorithms and vectorization
- **Pandas/NumPy**: Data processing and analysis

### Database Schema
```
employees (id, name, department, role, experience_years, salary, location, career_aspirations)
skills (id, skill_name, category, description)
employee_skills (employee_id, skill_id, proficiency_level, years_experience)
projects (id, name, description, priority, duration, budget, status)
project_skills (project_id, skill_id, importance_level, required_proficiency)
development_paths (id, name, description, duration, target_roles)
matching_results (employee_id, project_id, match_score, analysis, recommendations)
```

### AI Algorithms

#### 1. Skill Matching Algorithm
```python
def calculate_skill_match_score(employee_id, project_id):
    # Get employee skills and project requirements
    # Calculate weighted compatibility score
    # Apply proficiency level adjustments
    # Return normalized score (0-1)
```

#### 2. Experience Compatibility
```python
def calculate_experience_compatibility(employee_id, project_id):
    # Analyze years of experience vs project complexity
    # Apply duration and priority adjustments
    # Factor in role seniority requirements
    # Return compatibility score
```

#### 3. Career Alignment Analysis
```python
def calculate_career_alignment_score(employee_id, project_id):
    # Semantic similarity between career goals and project
    # Role keyword overlap analysis
    # Growth opportunity assessment
    # Return alignment score
```

#### 4. Comprehensive Matching
```python
final_score = (
    skill_score * 0.45 +
    experience_score * 0.25 +
    career_score * 0.20 +
    availability_score * 0.10
)
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip package manager
- SQLite (included with Python)

### Step 1: Clone and Setup
```bash
# Create project directory
mkdir hr-talent-matching
cd hr-talent-matching

# Copy all project files to this directory
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Initialize Database
```bash
python app.py
```
This will:
- Create SQLite database
- Set up all tables
- Load sample data
- Start the Flask server

### Step 4: Access the Application
Open your browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

### Dashboard Overview
- View system statistics and metrics
- Access quick actions for common tasks
- Monitor AI algorithm performance

### Employee Management
1. **View Employees**: Browse all employee profiles
2. **Find Projects**: Click "Find Projects" to get AI recommendations
3. **Skill Analysis**: View detailed skill breakdowns and gaps

### Project Management
1. **View Projects**: Browse all active and planned projects
2. **Find Team Members**: Click "Find Team Members" for AI matching
3. **Priority Management**: Filter by project priority and status

### AI Matching Process
1. **Employee → Projects**: Find best project matches for an employee
2. **Project → Employees**: Find best employee matches for a project
3. **Skill Gap Analysis**: Detailed breakdown of strengths and gaps
4. **Development Recommendations**: Personalized learning paths

## API Endpoints

### GET /api/match-score
Get comprehensive matching score between employee and project
```
Parameters: employee_id, project_id
Response: {overall_score, skill_match, experience_compatibility, career_alignment, ...}
```

### GET /api/skill-gap/{employee_id}/{project_id}
Get detailed skill gap analysis
```
Response: {strengths: [...], gaps: [...], gap_count, strength_count}
```

## Algorithm Details

### Skill Matching Process
1. **Data Extraction**: Extract employee skills and project requirements
2. **Proficiency Mapping**: Map skill levels to compatibility scores
3. **Importance Weighting**: Apply project-specific skill importance
4. **Gap Calculation**: Identify missing or insufficient skills
5. **Score Normalization**: Convert to 0-1 scale for consistency

### Text Analysis (TF-IDF + Cosine Similarity)
```python
# Career aspirations vs project description
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform([career_text, project_text])
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
```

### Development Path Recommendation
1. **Current Role Analysis**: Assess current position and skills
2. **Career Goal Parsing**: Extract target roles and aspirations
3. **Path Mapping**: Match to available development tracks
4. **Duration Estimation**: Calculate realistic timeframes
5. **Resource Suggestion**: Recommend training materials

## Testing

### Run Unit Tests
```bash
python test_matching.py
```

### Test Coverage
- Skill matching calculations
- Experience compatibility
- Comprehensive scoring
- API endpoint functionality
- Database operations

## Deployment Considerations

### Production Deployment
1. **Environment Variables**: Set production database URL and secret key
2. **Database Migration**: Use PostgreSQL or MySQL for production
3. **Caching**: Implement Redis for improved performance
4. **Load Balancing**: Use Gunicorn with multiple workers
5. **Monitoring**: Add logging and performance monitoring

### Security Features
- Input validation for all API endpoints
- SQL injection prevention via SQLAlchemy ORM
- XSS protection in templates
- CSRF protection (can be added with Flask-WTF)

### Scalability
- **Database Optimization**: Add indexes for faster queries
- **Caching Layer**: Cache matching results for frequent requests
- **Async Processing**: Background job processing for heavy calculations
- **Microservices**: Split into separate matching and web services

## Future Enhancements

### Advanced AI Features
- **Deep Learning Models**: Neural networks for more sophisticated matching
- **Natural Language Processing**: Advanced text analysis for job descriptions
- **Predictive Analytics**: Forecast employee career trajectories
- **Automated Resume Parsing**: Extract skills from resume documents

### Integration Capabilities
- **HR Systems**: Connect with existing HRIS platforms
- **Calendar Integration**: Real-time availability checking
- **Learning Management**: Direct integration with training platforms
- **Performance Tracking**: Connect with performance management systems

### Mobile Application
- **React Native App**: Mobile interface for managers and employees
- **Push Notifications**: Real-time matching alerts
- **Offline Capability**: Basic functionality without internet

## Contributing

### Code Structure
- `app.py`: Main Flask application and matching engine
- `templates/`: HTML templates for web interface  
- `config.py`: Configuration settings
- `api_utils.py`: API utilities and validation
- `db_utils.py`: Database utilities and backup functions
- `test_matching.py`: Unit test suite

### Development Workflow
1. Create feature branch
2. Implement changes with tests
3. Run test suite
4. Submit pull request
5. Code review and merge

## License

This project is developed for educational and hackathon purposes. Feel free to use and modify as needed.

## Support

For technical support or questions:
- Review the code documentation
- Run the test suite to verify functionality
- Check API endpoints with curl or Postman
- Examine sample data structure in CSV files

## Hackathon Demo

### Live Demo Flow
1. **Dashboard Overview** - Show system statistics
2. **Employee Matching** - Demonstrate AI recommendations
3. **Project Staffing** - Show team formation suggestions
4. **Skill Gap Analysis** - Display detailed breakdowns
5. **Development Paths** - Present career growth options
6. **API Integration** - Show real-time scoring

### Key Talking Points
- **AI-Driven Intelligence**: Advanced algorithms, not just keyword matching
- **Comprehensive Analysis**: Multi-factor scoring with detailed explanations
- **Practical Applications**: Real-world HR problems with actionable solutions
- **Scalable Architecture**: Production-ready design with clear expansion paths
- **User Experience**: Intuitive interface with rich visualizations

This system demonstrates the power of AI in transforming traditional HR processes into data-driven, intelligent solutions that benefit both employees and organizations.
