// Sample Data
const employeesData = [
    {
        id: 1,
        name: "Alice Johnson",
        department: "Engineering",
        role: "Senior Developer",
        experience_years: 5,
        current_salary: 90000,
        location: "New York",
        career_aspirations: "Tech Lead",
        skills: [
            { skill: "Python", proficiency: 4 },
            { skill: "SQL", proficiency: 4 },
            { skill: "AWS", proficiency: 3 },
            { skill: "Git", proficiency: 5 },
            { skill: "JavaScript", proficiency: 3 },
            { skill: "Problem Solving", proficiency: 4 }
        ]
    },
    {
        id: 2,
        name: "Bob Smith",
        department: "Marketing",
        role: "Marketing Manager",
        experience_years: 7,
        current_salary: 75000,
        location: "San Francisco",
        career_aspirations: "VP Marketing",
        skills: [
            { skill: "Marketing Strategy", proficiency: 5 },
            { skill: "SEO", proficiency: 4 },
            { skill: "Content Creation", proficiency: 4 },
            { skill: "Leadership", proficiency: 4 },
            { skill: "Communication", proficiency: 5 }
        ]
    },
    {
        id: 3,
        name: "Carol Davis",
        department: "Data Science",
        role: "Data Scientist",
        experience_years: 4,
        current_salary: 85000,
        location: "Boston",
        career_aspirations: "Senior Data Scientist",
        skills: [
            { skill: "Python", proficiency: 5 },
            { skill: "Machine Learning", proficiency: 4 },
            { skill: "Data Analysis", proficiency: 5 },
            { skill: "TensorFlow", proficiency: 3 },
            { skill: "SQL", proficiency: 4 },
            { skill: "Critical Thinking", proficiency: 5 }
        ]
    },
    {
        id: 4,
        name: "David Wilson",
        department: "Engineering",
        role: "DevOps Engineer",
        experience_years: 6,
        current_salary: 88000,
        location: "Seattle",
        career_aspirations: "Cloud Architect",
        skills: [
            { skill: "AWS", proficiency: 5 },
            { skill: "Docker", proficiency: 4 },
            { skill: "Kubernetes", proficiency: 4 },
            { skill: "Python", proficiency: 3 },
            { skill: "Problem Solving", proficiency: 5 }
        ]
    },
    {
        id: 5,
        name: "Emma Brown",
        department: "Product",
        role: "Product Manager",
        experience_years: 8,
        current_salary: 95000,
        location: "Austin",
        career_aspirations: "VP Product",
        skills: [
            { skill: "Project Management", proficiency: 5 },
            { skill: "Leadership", proficiency: 4 },
            { skill: "Communication", proficiency: 5 },
            { skill: "Critical Thinking", proficiency: 4 },
            { skill: "Team Collaboration", proficiency: 5 }
        ]
    }
];

const projectsData = [
    {
        id: 1,
        project_name: "AI-Powered Customer Service Bot",
        description: "Build an intelligent chatbot for customer support using NLP and ML",
        priority: "High",
        estimated_duration_months: 6,
        budget: 150000,
        status: "Planning",
        required_skills: [
            { skill: "Python", importance: 5, required_level: 4 },
            { skill: "Machine Learning", importance: 5, required_level: 4 },
            { skill: "TensorFlow", importance: 4, required_level: 3 },
            { skill: "Problem Solving", importance: 4, required_level: 4 }
        ]
    },
    {
        id: 2,
        project_name: "Mobile E-commerce Platform",
        description: "Develop a React Native mobile app for e-commerce with real-time features",
        priority: "Medium",
        estimated_duration_months: 4,
        budget: 100000,
        status: "Active",
        required_skills: [
            { skill: "JavaScript", importance: 5, required_level: 4 },
            { skill: "React", importance: 5, required_level: 4 },
            { skill: "Node.js", importance: 4, required_level: 3 },
            { skill: "Git", importance: 3, required_level: 3 }
        ]
    },
    {
        id: 3,
        project_name: "Data Analytics Dashboard",
        description: "Create interactive dashboards for business intelligence and analytics",
        priority: "High",
        estimated_duration_months: 3,
        budget: 75000,
        status: "Planning",
        required_skills: [
            { skill: "SQL", importance: 5, required_level: 4 },
            { skill: "Data Analysis", importance: 5, required_level: 4 },
            { skill: "Python", importance: 4, required_level: 3 },
            { skill: "Critical Thinking", importance: 4, required_level: 4 }
        ]
    },
    {
        id: 4,
        project_name: "Cloud Infrastructure Migration",
        description: "Migrate legacy systems to AWS cloud infrastructure",
        priority: "Medium",
        estimated_duration_months: 5,
        budget: 120000,
        status: "Active",
        required_skills: [
            { skill: "AWS", importance: 5, required_level: 4 },
            { skill: "Docker", importance: 4, required_level: 3 },
            { skill: "Kubernetes", importance: 4, required_level: 3 },
            { skill: "Problem Solving", importance: 4, required_level: 4 }
        ]
    }
];

const skillsCategories = {
    "Technical": ["Python", "JavaScript", "React", "Machine Learning", "SQL", "AWS", "Docker", "Kubernetes", "Data Analysis", "TensorFlow", "Node.js", "Git"],
    "Soft Skills": ["Project Management", "Leadership", "Communication", "Problem Solving", "Critical Thinking", "Team Collaboration"],
    "Marketing": ["Marketing Strategy", "SEO", "Content Creation"],
    "Design": ["UI/UX Design", "Figma", "Adobe Creative Suite"]
};

const developmentPaths = [
    {
        id: 1,
        path_name: "Machine Learning Engineer Track",
        description: "Advanced ML and AI development skills for senior technical roles",
        duration_months: 8,
        target_roles: ["Senior ML Engineer", "AI Research Lead"]
    },
    {
        id: 2,
        path_name: "Full-Stack Developer Track",
        description: "Comprehensive full-stack development including modern frameworks",
        duration_months: 6,
        target_roles: ["Senior Full-Stack Developer", "Tech Lead"]
    },
    {
        id: 3,
        path_name: "Cloud Architecture Track",
        description: "Enterprise cloud solutions and infrastructure management",
        duration_months: 7,
        target_roles: ["Cloud Architect", "Principal Engineer"]
    }
];

const matchingWeights = {
    skill_match: 0.45,
    experience: 0.25,
    career_alignment: 0.20,
    availability: 0.10
};

// Global Variables
let currentEmployees = [...employeesData];
let currentProjects = [...projectsData];
let skillsChart, projectsChart;

// Navigation Functions
function showSection(sectionName) {
    // Hide all sections
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => section.classList.remove('active'));
    
    // Show selected section
    const targetSection = document.getElementById(sectionName);
    if (targetSection) {
        targetSection.classList.add('active');
    }
    
    // Update navigation buttons
    const navBtns = document.querySelectorAll('.nav-btn');
    navBtns.forEach(btn => btn.classList.remove('active'));
    
    // Find the clicked button and make it active
    const clickedBtn = Array.from(navBtns).find(btn => btn.textContent.toLowerCase() === sectionName);
    if (clickedBtn) {
        clickedBtn.classList.add('active');
    }
    
    // Load section-specific content
    switch(sectionName) {
        case 'dashboard':
            setTimeout(() => loadDashboard(), 100);
            break;
        case 'employees':
            setTimeout(() => loadEmployees(), 100);
            break;
        case 'projects':
            setTimeout(() => loadProjects(), 100);
            break;
        case 'skills':
            setTimeout(() => loadSkills(), 100);
            break;
    }
}

// AI Matching Algorithm
function calculateMatch(employee, project) {
    const skillScore = calculateSkillMatch(employee, project);
    const experienceScore = calculateExperienceMatch(employee, project);
    const careerScore = calculateCareerAlignment(employee, project);
    const availabilityScore = 0.8; // Simplified availability score
    
    const totalScore = (
        skillScore * matchingWeights.skill_match +
        experienceScore * matchingWeights.experience +
        careerScore * matchingWeights.career_alignment +
        availabilityScore * matchingWeights.availability
    ) * 100;
    
    return {
        total: Math.round(totalScore),
        skill_match: Math.round(skillScore * 100),
        experience_match: Math.round(experienceScore * 100),
        career_alignment: Math.round(careerScore * 100),
        availability: Math.round(availabilityScore * 100)
    };
}

function calculateSkillMatch(employee, project) {
    const employeeSkillMap = {};
    employee.skills.forEach(skill => {
        employeeSkillMap[skill.skill] = skill.proficiency;
    });
    
    let totalRequiredImportance = 0;
    let matchedImportance = 0;
    
    project.required_skills.forEach(requiredSkill => {
        totalRequiredImportance += requiredSkill.importance;
        const employeeProficiency = employeeSkillMap[requiredSkill.skill] || 0;
        
        if (employeeProficiency >= requiredSkill.required_level) {
            matchedImportance += requiredSkill.importance;
        } else if (employeeProficiency > 0) {
            // Partial match based on proficiency level
            matchedImportance += requiredSkill.importance * (employeeProficiency / requiredSkill.required_level);
        }
    });
    
    return totalRequiredImportance > 0 ? matchedImportance / totalRequiredImportance : 0;
}

function calculateExperienceMatch(employee, project) {
    const requiredExperience = project.estimated_duration_months > 4 ? 4 : 2;
    const experienceRatio = Math.min(employee.experience_years / requiredExperience, 2);
    return Math.min(experienceRatio / 2, 1);
}

function calculateCareerAlignment(employee, project) {
    const careerKeywords = {
        "Tech Lead": ["AI", "Machine Learning", "Python", "Leadership"],
        "VP Marketing": ["Marketing", "Strategy", "Leadership"],
        "Senior Data Scientist": ["Machine Learning", "Data Analysis", "Python"],
        "Cloud Architect": ["AWS", "Docker", "Kubernetes", "Infrastructure"],
        "VP Product": ["Product", "Management", "Leadership", "Strategy"]
    };
    
    const aspirationKeywords = careerKeywords[employee.career_aspirations] || [];
    const projectText = `${project.project_name} ${project.description}`.toLowerCase();
    
    let matches = 0;
    aspirationKeywords.forEach(keyword => {
        if (projectText.includes(keyword.toLowerCase())) {
            matches++;
        }
    });
    
    return aspirationKeywords.length > 0 ? matches / aspirationKeywords.length : 0.5;
}

// Dashboard Functions
function loadDashboard() {
    updateStats();
    setTimeout(() => createCharts(), 200);
}

function updateStats() {
    document.getElementById('totalEmployees').textContent = employeesData.length;
    document.getElementById('totalProjects').textContent = projectsData.length;
    
    // Calculate unique skills
    const allSkills = new Set();
    employeesData.forEach(emp => {
        emp.skills.forEach(skill => allSkills.add(skill.skill));
    });
    document.getElementById('totalSkills').textContent = allSkills.size;
    
    // Calculate average match score
    let totalMatches = 0;
    let matchCount = 0;
    employeesData.forEach(emp => {
        projectsData.forEach(proj => {
            totalMatches += calculateMatch(emp, proj).total;
            matchCount++;
        });
    });
    const avgMatch = matchCount > 0 ? Math.round(totalMatches / matchCount) : 0;
    document.getElementById('avgMatchScore').textContent = `${avgMatch}%`;
}

function createCharts() {
    createSkillsChart();
    createProjectsChart();
}

function createSkillsChart() {
    const ctx = document.getElementById('skillsChart');
    if (!ctx) return;
    
    if (skillsChart) {
        skillsChart.destroy();
    }
    
    const categories = Object.keys(skillsCategories);
    const counts = categories.map(cat => skillsCategories[cat].length);
    
    skillsChart = new Chart(ctx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: categories,
            datasets: [{
                data: counts,
                backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5'],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function createProjectsChart() {
    const ctx = document.getElementById('projectsChart');
    if (!ctx) return;
    
    if (projectsChart) {
        projectsChart.destroy();
    }
    
    const priorityCounts = projectsData.reduce((acc, project) => {
        acc[project.priority] = (acc[project.priority] || 0) + 1;
        return acc;
    }, {});
    
    projectsChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: Object.keys(priorityCounts),
            datasets: [{
                label: 'Number of Projects',
                data: Object.values(priorityCounts),
                backgroundColor: ['#DB4545', '#D2BA4C', '#5D878F'],
                borderColor: ['#DB4545', '#D2BA4C', '#5D878F'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

// Employee Functions
function loadEmployees() {
    renderEmployees(currentEmployees);
    setupEmployeeFilters();
}

function renderEmployees(employees) {
    const container = document.getElementById('employeesList');
    if (!container) return;
    
    container.innerHTML = '';
    
    employees.forEach(employee => {
        const employeeCard = createEmployeeCard(employee);
        container.appendChild(employeeCard);
    });
}

function createEmployeeCard(employee) {
    const card = document.createElement('div');
    card.className = 'employee-card';
    
    const initials = employee.name.split(' ').map(n => n[0]).join('');
    
    card.innerHTML = `
        <div class="employee-header">
            <div class="employee-avatar">${initials}</div>
            <div>
                <h3 class="employee-name">${employee.name}</h3>
                <p class="employee-role">${employee.role}</p>
            </div>
        </div>
        
        <div class="employee-details">
            <div class="detail-row">
                <span class="detail-label">Department:</span>
                <span class="detail-value">${employee.department}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Experience:</span>
                <span class="detail-value">${employee.experience_years} years</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Location:</span>
                <span class="detail-value">${employee.location}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Career Goal:</span>
                <span class="detail-value">${employee.career_aspirations}</span>
            </div>
        </div>
        
        <div class="skills-list">
            <div class="skills-title">Skills & Proficiency</div>
            <div class="skills-container">
                ${employee.skills.map(skill => `
                    <span class="skill-item">
                        ${skill.skill}
                        <span class="skill-proficiency">${skill.proficiency}/5</span>
                    </span>
                `).join('')}
            </div>
        </div>
        
        <div class="action-buttons">
            <button class="btn btn--primary" onclick="findProjectsForEmployee(${employee.id})">
                Find Best Projects
            </button>
        </div>
    `;
    
    return card;
}

function setupEmployeeFilters() {
    const searchInput = document.getElementById('employeeSearch');
    const departmentFilter = document.getElementById('departmentFilter');
    
    if (searchInput) {
        searchInput.addEventListener('input', filterEmployees);
    }
    if (departmentFilter) {
        departmentFilter.addEventListener('change', filterEmployees);
    }
}

function filterEmployees() {
    const searchInput = document.getElementById('employeeSearch');
    const departmentFilter = document.getElementById('departmentFilter');
    
    const searchTerm = searchInput ? searchInput.value.toLowerCase() : '';
    const selectedDepartment = departmentFilter ? departmentFilter.value : '';
    
    const filtered = employeesData.filter(employee => {
        const matchesSearch = employee.name.toLowerCase().includes(searchTerm) ||
                            employee.skills.some(skill => skill.skill.toLowerCase().includes(searchTerm));
        const matchesDepartment = !selectedDepartment || employee.department === selectedDepartment;
        
        return matchesSearch && matchesDepartment;
    });
    
    currentEmployees = filtered;
    renderEmployees(filtered);
}

function findProjectsForEmployee(employeeId) {
    const employee = employeesData.find(emp => emp.id === employeeId);
    if (!employee) return;
    
    const projectMatches = projectsData.map(project => {
        const matchScore = calculateMatch(employee, project);
        return {
            project,
            matchScore,
            skillGaps: analyzeSkillGaps(employee, project)
        };
    }).sort((a, b) => b.matchScore.total - a.matchScore.total);
    
    showEmployeeProjectModal(employee, projectMatches);
}

function analyzeSkillGaps(employee, project) {
    const employeeSkillMap = {};
    employee.skills.forEach(skill => {
        employeeSkillMap[skill.skill] = skill.proficiency;
    });
    
    const gaps = {
        strengths: [],
        gaps: [],
        missing: []
    };
    
    project.required_skills.forEach(requiredSkill => {
        const employeeProficiency = employeeSkillMap[requiredSkill.skill];
        
        if (!employeeProficiency) {
            gaps.missing.push(requiredSkill);
        } else if (employeeProficiency >= requiredSkill.required_level) {
            gaps.strengths.push({...requiredSkill, employeeProficiency});
        } else {
            gaps.gaps.push({...requiredSkill, employeeProficiency});
        }
    });
    
    return gaps;
}

// Project Functions
function loadProjects() {
    renderProjects(currentProjects);
    setupProjectFilters();
}

function renderProjects(projects) {
    const container = document.getElementById('projectsList');
    if (!container) return;
    
    container.innerHTML = '';
    
    projects.forEach(project => {
        const projectCard = createProjectCard(project);
        container.appendChild(projectCard);
    });
}

function createProjectCard(project) {
    const card = document.createElement('div');
    card.className = 'project-card';
    
    card.innerHTML = `
        <div class="project-header">
            <h3 class="project-title">${project.project_name}</h3>
            <p class="project-description">${project.description}</p>
        </div>
        
        <div class="project-meta">
            <div class="meta-item">
                <div class="meta-label">Priority</div>
                <div class="meta-value priority-${project.priority.toLowerCase()}">${project.priority}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Status</div>
                <div class="meta-value status-${project.status.toLowerCase()}">${project.status}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Duration</div>
                <div class="meta-value">${project.estimated_duration_months} months</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Budget</div>
                <div class="meta-value">$${project.budget.toLocaleString()}</div>
            </div>
        </div>
        
        <div class="required-skills">
            <div class="required-skills-title">Required Skills</div>
            <div class="skills-container">
                ${project.required_skills.map(skill => `
                    <span class="required-skill-item">
                        ${skill.skill}
                        <span class="skill-importance">Level ${skill.required_level}</span>
                    </span>
                `).join('')}
            </div>
        </div>
        
        <div class="action-buttons">
            <button class="btn btn--primary" onclick="findTeamForProject(${project.id})">
                Find Best Team
            </button>
        </div>
    `;
    
    return card;
}

function setupProjectFilters() {
    const searchInput = document.getElementById('projectSearch');
    const priorityFilter = document.getElementById('priorityFilter');
    
    if (searchInput) {
        searchInput.addEventListener('input', filterProjects);
    }
    if (priorityFilter) {
        priorityFilter.addEventListener('change', filterProjects);
    }
}

function filterProjects() {
    const searchInput = document.getElementById('projectSearch');
    const priorityFilter = document.getElementById('priorityFilter');
    
    const searchTerm = searchInput ? searchInput.value.toLowerCase() : '';
    const selectedPriority = priorityFilter ? priorityFilter.value : '';
    
    const filtered = projectsData.filter(project => {
        const matchesSearch = project.project_name.toLowerCase().includes(searchTerm) ||
                            project.description.toLowerCase().includes(searchTerm);
        const matchesPriority = !selectedPriority || project.priority === selectedPriority;
        
        return matchesSearch && matchesPriority;
    });
    
    currentProjects = filtered;
    renderProjects(filtered);
}

function findTeamForProject(projectId) {
    const project = projectsData.find(proj => proj.id === projectId);
    if (!project) return;
    
    const employeeMatches = employeesData.map(employee => {
        const matchScore = calculateMatch(employee, project);
        return {
            employee,
            matchScore,
            skillGaps: analyzeSkillGaps(employee, project)
        };
    }).sort((a, b) => b.matchScore.total - a.matchScore.total);
    
    showProjectTeamModal(project, employeeMatches);
}

// Skills Functions
function loadSkills() {
    Object.keys(skillsCategories).forEach(category => {
        const containerMap = {
            'Technical': 'technicalSkills',
            'Soft Skills': 'softSkills',
            'Marketing': 'marketingSkills',
            'Design': 'designSkills'
        };
        
        const containerId = containerMap[category];
        const container = document.getElementById(containerId);
        
        if (container) {
            container.innerHTML = '';
            
            skillsCategories[category].forEach(skill => {
                const skillTag = document.createElement('span');
                skillTag.className = 'skill-tag';
                skillTag.textContent = skill;
                container.appendChild(skillTag);
            });
        }
    });
}

// Modal Functions
function showEmployeeProjectModal(employee, projectMatches) {
    const modal = document.getElementById('employeeMatchModal');
    const employeeInfo = document.getElementById('modalEmployeeInfo');
    const recommendations = document.getElementById('projectRecommendations');
    
    if (!modal || !employeeInfo || !recommendations) return;
    
    employeeInfo.innerHTML = `
        <h3>${employee.name}</h3>
        <p><strong>Role:</strong> ${employee.role} | <strong>Department:</strong> ${employee.department}</p>
        <p><strong>Experience:</strong> ${employee.experience_years} years | <strong>Career Goal:</strong> ${employee.career_aspirations}</p>
    `;
    
    recommendations.innerHTML = projectMatches.map(match => `
        <div class="recommendation-item">
            <div class="match-score ${getMatchScoreClass(match.matchScore.total)}">${match.matchScore.total}% Match</div>
            <div class="recommendation-header">
                <h4 class="recommendation-title">${match.project.project_name}</h4>
                <p class="recommendation-subtitle">${match.project.description}</p>
            </div>
            
            <div class="match-factors">
                <div class="factor">
                    <div class="factor-label">Skill Match</div>
                    <div class="factor-value">${match.matchScore.skill_match}%</div>
                </div>
                <div class="factor">
                    <div class="factor-label">Experience</div>
                    <div class="factor-value">${match.matchScore.experience_match}%</div>
                </div>
                <div class="factor">
                    <div class="factor-label">Career Alignment</div>
                    <div class="factor-value">${match.matchScore.career_alignment}%</div>
                </div>
                <div class="factor">
                    <div class="factor-label">Availability</div>
                    <div class="factor-value">${match.matchScore.availability}%</div>
                </div>
            </div>
            
            <div class="skill-gaps">
                <div class="skill-gaps-title">Skill Analysis</div>
                <div class="gap-analysis">
                    ${match.skillGaps.strengths.map(skill => 
                        `<span class="skill-status skill-strength">✓ ${skill.skill}</span>`
                    ).join('')}
                    ${match.skillGaps.gaps.map(skill => 
                        `<span class="skill-status skill-gap">△ ${skill.skill} (${skill.employeeProficiency}/${skill.required_level})</span>`
                    ).join('')}
                    ${match.skillGaps.missing.map(skill => 
                        `<span class="skill-status skill-missing">✗ ${skill.skill} (0/${skill.required_level})</span>`
                    ).join('')}
                </div>
            </div>
            
            ${match.skillGaps.gaps.length > 0 || match.skillGaps.missing.length > 0 ? `
                <div class="development-path">
                    <div class="development-title">Recommended Development Path</div>
                    <p>Focus on improving: ${[...match.skillGaps.gaps, ...match.skillGaps.missing].map(s => s.skill).join(', ')}</p>
                    <div class="development-details">
                        <span>Estimated Duration: 2-4 months</span>
                        <span>Success Rate: 85%</span>
                    </div>
                </div>
            ` : ''}
            
            <div class="action-buttons">
                <button class="btn btn--primary" onclick="assignToProject(${employee.id}, ${match.project.id})">Assign to Project</button>
                <button class="btn btn--outline" onclick="showSkillGapAnalysis(${employee.id}, ${match.project.id})">View Skill Gap Analysis</button>
            </div>
        </div>
    `).join('');
    
    modal.classList.remove('hidden');
}

function showProjectTeamModal(project, employeeMatches) {
    const modal = document.getElementById('projectMatchModal');
    const projectInfo = document.getElementById('modalProjectInfo');
    const recommendations = document.getElementById('employeeRecommendations');
    
    if (!modal || !projectInfo || !recommendations) return;
    
    projectInfo.innerHTML = `
        <h3>${project.project_name}</h3>
        <p>${project.description}</p>
        <p><strong>Priority:</strong> ${project.priority} | <strong>Duration:</strong> ${project.estimated_duration_months} months | <strong>Budget:</strong> $${project.budget.toLocaleString()}</p>
    `;
    
    recommendations.innerHTML = employeeMatches.map(match => `
        <div class="recommendation-item">
            <div class="match-score ${getMatchScoreClass(match.matchScore.total)}">${match.matchScore.total}% Match</div>
            <div class="recommendation-header">
                <h4 class="recommendation-title">${match.employee.name}</h4>
                <p class="recommendation-subtitle">${match.employee.role} - ${match.employee.department}</p>
            </div>
            
            <div class="match-factors">
                <div class="factor">
                    <div class="factor-label">Skill Match</div>
                    <div class="factor-value">${match.matchScore.skill_match}%</div>
                </div>
                <div class="factor">
                    <div class="factor-label">Experience</div>
                    <div class="factor-value">${match.matchScore.experience_match}%</div>
                </div>
                <div class="factor">
                    <div class="factor-label">Career Alignment</div>
                    <div class="factor-value">${match.matchScore.career_alignment}%</div>
                </div>
                <div class="factor">
                    <div class="factor-label">Availability</div>
                    <div class="factor-value">${match.matchScore.availability}%</div>
                </div>
            </div>
            
            <div class="skill-gaps">
                <div class="skill-gaps-title">Skill Coverage</div>
                <div class="gap-analysis">
                    ${match.skillGaps.strengths.map(skill => 
                        `<span class="skill-status skill-strength">✓ ${skill.skill}</span>`
                    ).join('')}
                    ${match.skillGaps.gaps.map(skill => 
                        `<span class="skill-status skill-gap">△ ${skill.skill}</span>`
                    ).join('')}
                    ${match.skillGaps.missing.map(skill => 
                        `<span class="skill-status skill-missing">✗ ${skill.skill}</span>`
                    ).join('')}
                </div>
            </div>
            
            <div class="action-buttons">
                <button class="btn btn--primary" onclick="assignToProject(${match.employee.id}, ${project.id})">Add to Team</button>
                <button class="btn btn--outline" onclick="showSkillGapAnalysis(${match.employee.id}, ${project.id})">View Development Plan</button>
            </div>
        </div>
    `).join('');
    
    modal.classList.remove('hidden');
}

function showSkillGapAnalysis(employeeId, projectId) {
    const employee = employeesData.find(emp => emp.id === employeeId);
    const project = projectsData.find(proj => proj.id === projectId);
    const skillGaps = analyzeSkillGaps(employee, project);
    
    const modal = document.getElementById('skillGapModal');
    const content = document.getElementById('skillGapContent');
    
    if (!modal || !content) return;
    
    content.innerHTML = `
        <div class="employee-project-info">
            <h3>${employee.name} → ${project.project_name}</h3>
            <p>Detailed skill gap analysis and personalized development recommendations</p>
        </div>
        
        <div class="gap-analysis-detailed">
            <h4>Skill Strengths (${skillGaps.strengths.length})</h4>
            <div class="skills-breakdown">
                ${skillGaps.strengths.map(skill => `
                    <div class="skill-detail strength">
                        <span class="skill-name">${skill.skill}</span>
                        <span class="skill-levels">Employee: ${skill.employeeProficiency}/5 | Required: ${skill.required_level}/5</span>
                        <span class="skill-status-text">✓ Exceeds Requirements</span>
                    </div>
                `).join('')}
            </div>
            
            <h4>Skills Needing Improvement (${skillGaps.gaps.length})</h4>
            <div class="skills-breakdown">
                ${skillGaps.gaps.map(skill => `
                    <div class="skill-detail gap">
                        <span class="skill-name">${skill.skill}</span>
                        <span class="skill-levels">Employee: ${skill.employeeProficiency}/5 | Required: ${skill.required_level}/5</span>
                        <span class="skill-status-text">△ Gap: ${skill.required_level - skill.employeeProficiency} level(s)</span>
                    </div>
                `).join('')}
            </div>
            
            <h4>Missing Skills (${skillGaps.missing.length})</h4>
            <div class="skills-breakdown">
                ${skillGaps.missing.map(skill => `
                    <div class="skill-detail missing">
                        <span class="skill-name">${skill.skill}</span>
                        <span class="skill-levels">Employee: 0/5 | Required: ${skill.required_level}/5</span>
                        <span class="skill-status-text">✗ Needs to learn</span>
                    </div>
                `).join('')}
            </div>
        </div>
        
        <div class="development-recommendations">
            <h4>Personalized Development Plan</h4>
            ${generateDevelopmentPlan(skillGaps)}
        </div>
    `;
    
    modal.classList.remove('hidden');
}

function generateDevelopmentPlan(skillGaps) {
    const gapsAndMissing = [...skillGaps.gaps, ...skillGaps.missing];
    
    if (gapsAndMissing.length === 0) {
        return '<p>🎉 Excellent! This employee has all the required skills for this project.</p>';
    }
    
    const developmentItems = gapsAndMissing.map(skill => {
        const isNewSkill = skill.employeeProficiency === undefined;
        const timeEstimate = isNewSkill ? '6-8 weeks' : '3-4 weeks';
        const resources = getSkillResources(skill.skill);
        
        return `
            <div class="development-item">
                <div class="development-header">
                    <h5>${skill.skill}</h5>
                    <span class="development-timeline">${timeEstimate}</span>
                </div>
                <div class="development-content">
                    <p><strong>Current Level:</strong> ${skill.employeeProficiency || 0}/5 | <strong>Target Level:</strong> ${skill.required_level}/5</p>
                    <p><strong>Recommended Resources:</strong> ${resources}</p>
                    <p><strong>Success Rate:</strong> 90% (Based on similar employee profiles)</p>
                </div>
            </div>
        `;
    }).join('');
    
    return `
        <div class="development-overview">
            <p><strong>Estimated Total Duration:</strong> 8-12 weeks</p>
            <p><strong>Recommended Learning Path:</strong> Start with foundational skills, then advance to specialized areas</p>
        </div>
        <div class="development-items">
            ${developmentItems}
        </div>
    `;
}

function getSkillResources(skill) {
    const resources = {
        'Python': 'Python.org tutorials, Codecademy Python course, Real Python',
        'JavaScript': 'MDN Web Docs, freeCodeCamp, JavaScript.info',
        'Machine Learning': 'Coursera ML course, Kaggle Learn, Hands-On Machine Learning book',
        'AWS': 'AWS Training, A Cloud Guru, AWS Solutions Architect certification',
        'Docker': 'Docker documentation, Docker Mastery course, Play with Docker',
        'SQL': 'SQLBolt, W3Schools SQL, PostgreSQL tutorial',
        'React': 'React documentation, React tutorial, Scrimba React course',
        'TensorFlow': 'TensorFlow tutorials, Deep Learning Specialization, TensorFlow certification'
    };
    
    return resources[skill] || 'Industry-standard courses, documentation, and hands-on projects';
}

function getMatchScoreClass(score) {
    if (score >= 80) return 'match-excellent';
    if (score >= 60) return 'match-good';
    return 'match-fair';
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('hidden');
    }
}

function assignToProject(employeeId, projectId) {
    const employee = employeesData.find(emp => emp.id === employeeId);
    const project = projectsData.find(proj => proj.id === projectId);
    
    if (employee && project) {
        alert(`✅ ${employee.name} has been assigned to "${project.project_name}"!`);
        closeModal('employeeMatchModal');
        closeModal('projectMatchModal');
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Set up navigation event listeners
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach((btn, index) => {
        const sections = ['dashboard', 'employees', 'projects', 'skills'];
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            showSection(sections[index]);
        });
    });
    
    // Initialize with dashboard
    showSection('dashboard');
    
    // Close modals when clicking outside
    document.addEventListener('click', function(event) {
        if (event.target.classList.contains('modal')) {
            event.target.classList.add('hidden');
        }
    });
    
    // Handle escape key for modals
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            const openModals = document.querySelectorAll('.modal:not(.hidden)');
            openModals.forEach(modal => modal.classList.add('hidden'));
        }
    });
});