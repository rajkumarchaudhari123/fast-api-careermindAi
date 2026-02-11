from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from fastapi.responses import FileResponse, JSONResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid

load_dotenv()

# ========================================
# FASTAPI APP WITH CORS
# ========================================

app = FastAPI(title="CareerMind AI Backend", version="3.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, production me specific domains daalo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# 🔥 PYTORCH CAREER PREDICTION MODEL
# ========================================

class CareerModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_careers=12):
        super(CareerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_careers)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = CareerModel()
model.eval()

# ========================================
# 📊 CAREER PATHS MAPPING (12 Careers)
# ========================================

CAREER_PATHS = {
    0: {
        "title": "Machine Learning Engineer",
        "description": "AI/ML models banate hain, data se predictions karte hain. Artificial Intelligence aur Deep Learning mein specialize karte hain.",
        "skills": ["Python", "TensorFlow/PyTorch", "Statistics", "Linear Algebra", "Data Structures", "ML Algorithms"],
        "roadmap": [
            "1. Python + Mathematics (3 months)",
            "2. Machine Learning Basics (3 months)",
            "3. Deep Learning & Neural Networks (4 months)",
            "4. Computer Vision / NLP (3 months)",
            "5. Projects & Kaggle Competitions (2 months)",
            "6. Internship/Job Preparation (2 months)"
        ],
        "salary": "₹8-30 LPA (India) | $100k-250k (USA)"
    },
    1: {
        "title": "Full Stack Web Developer",
        "description": "Frontend aur backend dono handle karte hain. Complete web applications build karte hain.",
        "skills": ["HTML/CSS", "JavaScript", "React/Vue", "Node.js", "MongoDB/PostgreSQL", "API Development"],
        "roadmap": [
            "1. HTML/CSS/JavaScript (2 months)",
            "2. Frontend Framework (React) (3 months)",
            "3. Backend (Node.js/Python) (3 months)",
            "4. Database & API Design (2 months)",
            "5. Authentication & Security (2 months)",
            "6. Deployment & DevOps (2 months)"
        ],
        "salary": "₹6-25 LPA (India) | $80k-180k (USA)"
    },
    2: {
        "title": "UI/UX Designer",
        "description": "User-friendly designs banate hain. User experience aur interface design mein expert hote hain.",
        "skills": ["Figma", "Adobe XD", "User Research", "Prototyping", "Wireframing", "Design Systems"],
        "roadmap": [
            "1. Design Principles & Color Theory (1 month)",
            "2. Figma/Adobe XD Mastery (2 months)",
            "3. User Research Methods (2 months)",
            "4. Interaction Design (2 months)",
            "5. Portfolio Projects (3 months)",
            "6. Internship/Job Preparation (2 months)"
        ],
        "salary": "₹5-20 LPA (India) | $70k-160k (USA)"
    },
    3: {
        "title": "Tech Entrepreneur",
        "description": "Apni tech startup karte hain. Innovation aur business dono handle karte hain.",
        "skills": ["Business Strategy", "Leadership", "Product Management", "Marketing", "Fundraising", "Team Building"],
        "roadmap": [
            "1. Market Research & Idea Validation (2 months)",
            "2. MVP Development (3 months)",
            "3. Funding & Networking (2 months)",
            "4. Team Building (2 months)",
            "5. Product Launch & Marketing (2 months)",
            "6. Scale & Growth (Ongoing)"
        ],
        "salary": "Unlimited (Based on success) | Potential: Crores/ Millions"
    },
    4: {
        "title": "Data Scientist",
        "description": "Data se insights nikalte hain. Statistics, machine learning aur programming combine karte hain.",
        "skills": ["Python/R", "SQL", "Machine Learning", "Data Visualization", "Statistics", "Big Data"],
        "roadmap": [
            "1. Python + SQL (2 months)",
            "2. Statistics & Probability (2 months)",
            "3. Machine Learning (3 months)",
            "4. Data Visualization (2 months)",
            "5. Big Data Tools (2 months)",
            "6. Real-world Projects (3 months)"
        ],
        "salary": "₹7-28 LPA (India) | $90k-200k (USA)"
    },
    5: {
        "title": "Mobile App Developer",
        "description": "Android/iOS apps banate hain. Cross-platform ya native development karte hain.",
        "skills": ["React Native/Flutter", "Java/Kotlin/Swift", "API Integration", "State Management", "App Store Deployment"],
        "roadmap": [
            "1. JavaScript/Dart Basics (1 month)",
            "2. React Native/Flutter (3 months)",
            "3. State Management (1 month)",
            "4. Backend Integration (2 months)",
            "5. Testing & Debugging (1 month)",
            "6. App Store Deployment (1 month)"
        ],
        "salary": "₹6-22 LPA (India) | $80k-180k (USA)"
    },
    6: {
        "title": "DevOps Engineer",
        "description": "Deployment aur infrastructure manage karte hain. CI/CD pipelines aur cloud infrastructure.",
        "skills": ["Docker", "Kubernetes", "AWS/Azure", "CI/CD", "Linux", "Networking"],
        "roadmap": [
            "1. Linux & Networking (2 months)",
            "2. Cloud Basics (AWS/GCP) (2 months)",
            "3. Docker & Kubernetes (3 months)",
            "4. CI/CD Pipelines (2 months)",
            "5. Infrastructure as Code (2 months)",
            "6. Monitoring & Security (2 months)"
        ],
        "salary": "₹8-32 LPA (India) | $100k-220k (USA)"
    },
    7: {
        "title": "Cybersecurity Expert",
        "description": "Systems ko hackers se bachate hain. Network security aur ethical hacking.",
        "skills": ["Network Security", "Ethical Hacking", "Cryptography", "Penetration Testing", "Firewalls", "Security Auditing"],
        "roadmap": [
            "1. Networking Basics (2 months)",
            "2. Linux & Scripting (2 months)",
            "3. Security Fundamentals (3 months)",
            "4. Ethical Hacking (3 months)",
            "5. Security Tools (2 months)",
            "6. Certifications (CEH/OSCP) (2 months)"
        ],
        "salary": "₹7-35 LPA (India) | $90k-250k (USA)"
    },
    8: {
        "title": "Content Creator / YouTuber",
        "description": "Educational content banate hain. Tech tutorials, reviews aur career guidance.",
        "skills": ["Video Editing", "Script Writing", "Public Speaking", "SEO", "Social Media", "Content Strategy"],
        "roadmap": [
            "1. Niche Selection (1 month)",
            "2. Content Planning (1 month)",
            "3. Video/Audio Production (2 months)",
            "4. Editing Skills (2 months)",
            "5. Channel Growth (3 months)",
            "6. Monetization Strategies (2 months)"
        ],
        "salary": "₹5-50 LPA (India) | Variable (Ad revenue, sponsorships)"
    },
    9: {
        "title": "Digital Marketer",
        "description": "Online marketing campaigns manage karte hain. SEO, social media aur PPC.",
        "skills": ["SEO", "Social Media Marketing", "Google Analytics", "Content Marketing", "PPC", "Email Marketing"],
        "roadmap": [
            "1. Marketing Fundamentals (1 month)",
            "2. SEO Basics (2 months)",
            "3. Social Media Marketing (2 months)",
            "4. Google Ads/Analytics (2 months)",
            "5. Content Strategy (2 months)",
            "6. Campaign Management (2 months)"
        ],
        "salary": "₹4-20 LPA (India) | $60k-150k (USA)"
    },
    10: {
        "title": "Game Developer",
        "description": "Video games banate hain. Game design, programming aur graphics.",
        "skills": ["Unity/Unreal Engine", "C#/C++", "Game Physics", "3D Modeling", "Game Design", "Animation"],
        "roadmap": [
            "1. Programming Basics (1 month)",
            "2. Game Engine (Unity/Unreal) (3 months)",
            "3. Game Design Principles (2 months)",
            "4. Graphics & Animation (3 months)",
            "5. Game Physics (2 months)",
            "6. Portfolio Projects (3 months)"
        ],
        "salary": "₹5-25 LPA (India) | $70k-180k (USA)"
    },
    11: {
        "title": "Blockchain Developer",
        "description": "Blockchain applications banate hain. Smart contracts aur decentralized apps.",
        "skills": ["Solidity", "Ethereum", "Smart Contracts", "Web3.js", "Cryptography", "DApps"],
        "roadmap": [
            "1. Blockchain Fundamentals (1 month)",
            "2. Solidity Programming (2 months)",
            "3. Smart Contracts (2 months)",
            "4. DApp Development (2 months)",
            "5. Web3 Integration (2 months)",
            "6. Security & Testing (2 months)"
        ],
        "salary": "₹10-40 LPA (India) | $120k-300k (USA)"
    }
}

# ========================================
# 🎯 EXPANDED QUESTION TREE (25+ Options)
# ========================================

QUESTION_TREE = {
    "id": 1,
    "question": "Tum free time me kya karna pasand karte ho?",
    "level": 1,
    "options": {
        "Coding / Programming": {
            "next": {
                "id": 2,
                "question": "Kaunsi programming language pasand hai?",
                "level": 2,
                "options": {
                    "Python": {
                        "next": {
                            "id": 3,
                            "question": "Python me kya seekhna chahoge?",
                            "level": 3,
                            "options": {
                                "Machine Learning / AI": {"career_id": 0, "features": [1,0,0,0,1,0,0,0,0,0,0,0,1,0,0]},
                                "Data Science": {"career_id": 4, "features": [1,0,0,0,1,0,0,0,0,0,0,0,1,0,0]},
                                "Web Development (Django/Flask)": {"career_id": 1, "features": [1,0,0,0,0,1,0,0,0,0,0,0,1,0,0]},
                                "Automation / Scripting": {"career_id": 6, "features": [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0]},
                            }
                        }
                    },
                    "JavaScript": {
                        "next": {
                            "id": 4,
                            "question": "JavaScript me kya banna chahoge?",
                            "level": 3,
                            "options": {
                                "Frontend Developer (React/Vue)": {"career_id": 1, "features": [1,0,0,0,0,0,0,1,0,0,0,0,1,0,0]},
                                "Backend Developer (Node.js)": {"career_id": 1, "features": [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0]},
                                "Mobile App (React Native)": {"career_id": 5, "features": [1,0,0,0,0,0,0,0,0,1,0,0,1,0,0]},
                                "Full Stack Developer": {"career_id": 1, "features": [1,0,0,0,0,0,0,0,0,0,1,0,1,0,0]},
                            }
                        }
                    },
                    "Java / Kotlin": {
                        "next": {
                            "id": 5,
                            "question": "Kis field me jana chahoge?",
                            "level": 3,
                            "options": {
                                "Android App Development": {"career_id": 5, "features": [0,1,0,0,0,0,0,0,0,0,0,0,1,0,0]},
                                "Enterprise Software": {"career_id": 1, "features": [0,1,0,0,0,0,0,0,0,0,0,0,1,0,0]},
                                "Backend Systems": {"career_id": 6, "features": [0,1,0,0,0,0,1,0,0,0,0,0,1,0,0]},
                            }
                        }
                    },
                    "C++ / C#": {
                        "next": {
                            "id": 6,
                            "question": "Kya develop karna chahoge?",
                            "level": 3,
                            "options": {
                                "Game Development": {"career_id": 10, "features": [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0]},
                                "System Software": {"career_id": 6, "features": [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0]},
                                "High Performance Apps": {"career_id": 1, "features": [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]},
                            }
                        }
                    }
                }
            }
        },
        "Design / Art": {
            "next": {
                "id": 7,
                "question": "Kaunsa design pasand hai?",
                "level": 2,
                "options": {
                    "UI/UX Design": {
                        "next": {
                            "id": 8,
                            "question": "Kya specialize karna chahoge?",
                            "level": 3,
                            "options": {
                                "Mobile App Design": {"career_id": 2, "features": [0,0,1,0,0,0,0,1,0,0,0,0,0,1,0]},
                                "Web Design": {"career_id": 2, "features": [0,0,1,0,0,0,0,0,1,0,0,0,0,1,0]},
                                "Product Design": {"career_id": 2, "features": [0,0,1,0,1,0,0,0,0,0,0,0,0,1,0]},
                            }
                        }
                    },
                    "Graphic Design": {
                        "next": {
                            "id": 9,
                            "question": "Kahan use karna chahoge?",
                            "level": 3,
                            "options": {
                                "Marketing & Branding": {"career_id": 9, "features": [0,0,1,0,0,1,0,0,0,0,0,0,0,1,0]},
                                "Digital Art": {"career_id": 2, "features": [0,0,1,0,0,0,0,1,0,0,0,0,0,1,0]},
                                "Animation": {"career_id": 10, "features": [0,0,1,0,0,0,0,0,0,0,0,1,0,1,0]},
                            }
                        }
                    },
                    "Video Editing": {
                        "next": {
                            "id": 10,
                            "question": "Kis type ka content?",
                            "level": 3,
                            "options": {
                                "YouTube Videos": {"career_id": 8, "features": [0,0,1,0,0,0,0,0,0,1,0,0,0,1,0]},
                                "Short Films": {"career_id": 2, "features": [0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]},
                                "Commercial Ads": {"career_id": 9, "features": [0,0,1,0,0,1,0,0,0,0,0,0,0,1,0]},
                            }
                        }
                    }
                }
            }
        },
        "Business / Entrepreneurship": {
            "next": {
                "id": 11,
                "question": "Business me kya interest hai?",
                "level": 2,
                "options": {
                    "Tech Startup": {
                        "next": {
                            "id": 12,
                            "question": "Kis type ki startup?",
                            "level": 3,
                            "options": {
                                "SaaS Product": {"career_id": 3, "features": [0,0,0,1,0,0,1,0,0,0,0,0,0,0,1]},
                                "E-commerce": {"career_id": 3, "features": [0,0,0,1,0,0,0,1,0,0,0,0,0,0,1]},
                                "Mobile App Business": {"career_id": 3, "features": [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1]},
                                "EdTech Platform": {"career_id": 3, "features": [0,0,0,1,1,0,0,0,0,0,0,0,0,0,1]},
                            }
                        }
                    },
                    "Digital Marketing": {
                        "next": {
                            "id": 13,
                            "question": "Kis platform pe?",
                            "level": 3,
                            "options": {
                                "Social Media Marketing": {"career_id": 9, "features": [0,0,0,1,0,1,0,0,0,0,0,0,0,0,1]},
                                "SEO & Content": {"career_id": 9, "features": [0,0,0,1,0,1,0,0,0,0,0,0,0,0,1]},
                                "Influencer Marketing": {"career_id": 8, "features": [0,0,0,1,0,1,0,0,0,1,0,0,0,0,1]},
                            }
                        }
                    },
                    "Finance / Trading": {
                        "next": {
                            "id": 14,
                            "question": "Kis type ka finance?",
                            "level": 3,
                            "options": {
                                "Stock Market": {"career_id": 3, "features": [0,0,0,1,0,0,0,0,0,0,0,0,0,0,1]},
                                "Cryptocurrency": {"career_id": 11, "features": [0,0,0,1,0,0,0,0,0,0,0,1,0,0,1]},
                                "FinTech Startup": {"career_id": 3, "features": [0,0,0,1,0,0,1,0,0,0,0,0,0,0,1]},
                            }
                        }
                    }
                }
            }
        },
        "Teaching / Coaching": {
            "next": {
                "id": 15,
                "question": "Kya padhana chahoge?",
                "level": 2,
                "options": {
                    "Programming": {
                        "next": {
                            "id": 16,
                            "question": "Kis level pe?",
                            "level": 3,
                            "options": {
                                "Beginners (Python/JS)": {"career_id": 1, "features": [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0]},
                                "Advanced (ML/System Design)": {"career_id": 0, "features": [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0]},
                                "Competitive Programming": {"career_id": 1, "features": [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    },
                    "Design Skills": {
                        "next": {
                            "id": 17,
                            "question": "Kis format me?",
                            "level": 3,
                            "options": {
                                "Online Courses": {"career_id": 2, "features": [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0]},
                                "YouTube Tutorials": {"career_id": 8, "features": [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]},
                                "1-on-1 Mentorship": {"career_id": 2, "features": [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0]},
                            }
                        }
                    },
                    "Career Guidance": {
                        "next": {
                            "id": 18,
                            "question": "Kis field me?",
                            "level": 3,
                            "options": {
                                "Tech Careers": {"career_id": 8, "features": [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0]},
                                "Study Abroad": {"career_id": 9, "features": [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0]},
                                "Interview Preparation": {"career_id": 1, "features": [0,0,0,0,1,0,0,0,0,0,0,1,0,0,0]},
                            }
                        }
                    }
                }
            }
        },
        "Writing / Content Creation": {
            "next": {
                "id": 19,
                "question": "Kya likhna pasand hai?",
                "level": 2,
                "options": {
                    "Technical Writing": {
                        "next": {
                            "id": 20,
                            "question": "Kis field me?",
                            "level": 3,
                            "options": {
                                "Programming Tutorials": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Documentation": {"career_id": 1, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                                "Blogging": {"career_id": 9, "features": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]},
                            }
                        }
                    },
                    "Creative Writing": {
                        "next": {
                            "id": 21,
                            "question": "Kis medium pe?",
                            "level": 3,
                            "options": {
                                "Social Media Posts": {"career_id": 9, "features": [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]},
                                "Short Stories": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Script Writing": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    }
                }
            }
        },
        "Gaming": {
            "next": {
                "id": 22,
                "question": "Kis type ka gaming?",
                "level": 2,
                "options": {
                    "Game Development": {
                        "next": {
                            "id": 23,
                            "question": "Kya develop karna chahoge?",
                            "level": 3,
                            "options": {
                                "Mobile Games": {"career_id": 10, "features": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]},
                                "PC/Console Games": {"career_id": 10, "features": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]},
                                "VR/AR Games": {"career_id": 10, "features": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]},
                            }
                        }
                    },
                    "Game Streaming": {
                        "next": {
                            "id": 24,
                            "question": "Kis platform pe?",
                            "level": 3,
                            "options": {
                                "YouTube Gaming": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Twitch Streaming": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Esports": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    }
                }
            }
        },
        "Music / Audio": {
            "next": {
                "id": 25,
                "question": "Music me kya karna chahoge?",
                "level": 2,
                "options": {
                    "Music Production": {
                        "next": {
                            "id": 26,
                            "question": "Kis type ki music?",
                            "level": 3,
                            "options": {
                                "EDM / Electronic": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Film Music": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                                "Podcast Production": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]},
                            }
                        }
                    },
                    "Instrument Playing": {
                        "next": {
                            "id": 27,
                            "question": "Kya bajana chahoge?",
                            "level": 3,
                            "options": {
                                "Guitar / Piano": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Digital Music": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    }
                }
            }
        },
        "Photography / Videography": {
            "next": {
                "id": 28,
                "question": "Kis type ki photography?",
                "level": 2,
                "options": {
                    "Portrait Photography": {
                        "next": {
                            "id": 29,
                            "question": "Kahan use karna chahoge?",
                            "level": 3,
                            "options": {
                                "Social Media": {"career_id": 9, "features": [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]},
                                "Commercial Work": {"career_id": 9, "features": [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0]},
                                "Art Portfolio": {"career_id": 2, "features": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]},
                            }
                        }
                    },
                    "Travel Vlogging": {
                        "next": {
                            "id": 30,
                            "question": "Kis platform pe?",
                            "level": 3,
                            "options": {
                                "YouTube Vlogs": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Instagram Reels": {"career_id": 9, "features": [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]},
                                "Travel Blog": {"career_id": 9, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    }
                }
            }
        },
        "Fitness / Sports": {
            "next": {
                "id": 31,
                "question": "Fitness me kya interest hai?",
                "level": 2,
                "options": {
                    "Gym / Workout": {
                        "next": {
                            "id": 32,
                            "question": "Kya karna chahoge?",
                            "level": 3,
                            "options": {
                                "Personal Trainer": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Fitness Influencer": {"career_id": 8, "features": [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]},
                                "Nutrition Coach": {"career_id": 9, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    },
                    "Sports Coaching": {
                        "next": {
                            "id": 33,
                            "question": "Kaunsa sport?",
                            "level": 3,
                            "options": {
                                "Cricket / Football": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "eSports Coach": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    }
                }
            }
        },
        "Reading Books": {
            "next": {
                "id": 34,
                "question": "Kis type ki books padhte ho?",
                "level": 2,
                "options": {
                    "Fiction / Novels": {
                        "next": {
                            "id": 35,
                            "question": "Kya karna chahoge?",
                            "level": 3,
                            "options": {
                                "Author / Writer": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Book Reviewer": {"career_id": 9, "features": [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]},
                                "Content Creator": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    },
                    "Self-Help / Business": {
                        "next": {
                            "id": 36,
                            "question": "Kis field me apply karna chahoge?",
                            "level": 3,
                            "options": {
                                "Life Coach": {"career_id": 9, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Business Consultant": {"career_id": 3, "features": [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0]},
                                "Motivational Speaker": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]},
                            }
                        }
                    }
                }
            }
        },
        "Watching Movies/TV": {
            "next": {
                "id": 37,
                "question": "Kis type ki movies/series?",
                "level": 2,
                "options": {
                    "Tech / Sci-Fi": {
                        "next": {
                            "id": 38,
                            "question": "Kya karna chahoge?",
                            "level": 3,
                            "options": {
                                "Tech Reviewer": {"career_id": 8, "features": [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]},
                                "Film Critic": {"career_id": 9, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Content Creator": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    },
                    "Documentaries": {
                        "next": {
                            "id": 39,
                            "question": "Kis topic pe?",
                            "level": 3,
                            "options": {
                                "Science & Tech": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "History & Culture": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]},
                                "Nature & Wildlife": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]},
                            }
                        }
                    }
                }
            }
        },
        "Cooking / Baking": {
            "next": {
                "id": 40,
                "question": "Kya banana pasand hai?",
                "level": 2,
                "options": {
                    "Indian Cuisine": {
                        "next": {
                            "id": 41,
                            "question": "Kya karna chahoge?",
                            "level": 3,
                            "options": {
                                "Food Blogger": {"career_id": 9, "features": [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]},
                                "YouTube Chef": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Restaurant Business": {"career_id": 3, "features": [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    },
                    "International Cuisine": {
                        "next": {
                            "id": 42,
                            "question": "Kaunsa cuisine?",
                            "level": 3,
                            "options": {
                                "Italian / Chinese": {"career_id": 8, "features": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]},
                                "Bakery / Desserts": {"career_id": 3, "features": [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0]},
                            }
                        }
                    }
                }
            }
        }
    }
}

# ========================================
# 📊 DATABASE FUNCTIONS
# ========================================

def get_db_connection():
    """Create database connection with error handling"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "careermind_ai"),
            port=int(os.getenv("DB_PORT", "3306")),
            charset='utf8mb4'
        )
        print("✅ Database connection successful")
        return connection
    except Error as e:
        print(f"❌ Database connection error: {e}")
        return None

def setup_database():
    """Setup database and tables on startup"""
    try:
        # Connect without database first
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "")
        )
        
        cursor = connection.cursor()
        
        # Create database if not exists
        db_name = os.getenv("DB_NAME", "careermind_ai")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        print(f"✅ Database '{db_name}' ready")
        
        # Use the database
        cursor.execute(f"USE {db_name}")
        
        # Create user_careers table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS user_careers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_name VARCHAR(100) NOT NULL,
            selections TEXT NOT NULL,
            career_id INT NOT NULL,
            career_title VARCHAR(100) NOT NULL,
            confidence FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_user_name (user_name),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        cursor.execute(create_table_query)
        print("✅ Table 'user_careers' created/verified")
        
        cursor.close()
        connection.close()
        
    except Error as e:
        print(f"❌ Database setup error: {e}")

def save_user_career(user_name: str, selections: List[str], career_id: int, career_title: str, confidence: float):
    """Save user career data to database"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print("⚠️ Database connection failed, skipping save")
            return False
            
        cursor = conn.cursor()
        
        # Convert list to string
        selections_str = "|".join(selections)  # Using | as separator
        
        query = """
        INSERT INTO user_careers 
        (user_name, selections, career_id, career_title, confidence) 
        VALUES (%s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            user_name,
            selections_str,
            career_id,
            career_title,
            confidence
        ))
        
        conn.commit()
        cursor.close()
        
        print(f"✅ Saved to DB: {user_name} -> {career_title} ({confidence}%)")
        return True
        
    except Error as e:
        print(f"❌ Database save error: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# ========================================
# 🚀 API SCHEMAS
# ========================================

class AnswerRequest(BaseModel):
    user_name: str = Field(default="Anonymous", min_length=1)
    current_path: List[str] = Field(default_factory=list)
    question_id: Optional[int] = Field(default=1)

class FinalAnswer(BaseModel):
    user_name: str
    selections: List[str]
    career_id: int

# ========================================
# 🎯 API ENDPOINTS
# ========================================

@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    print("\n" + "="*50)
    print("🚀 Starting CareerMind AI Backend v3.0")
    print("="*50)
    setup_database()
    print("✅ Server startup complete")
    print("="*50 + "\n")

@app.get("/")
async def home():
    return {
        "message": "🎯 CareerMind AI Backend",
        "version": "3.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/health",
            "/question",
            "/answer (POST)",
            "/generate-pdf (POST)",
            "/download-pdf/{filename}",
            "/stats"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = "healthy" if get_db_connection() else "unhealthy"
    return {
        "status": "healthy",
        "database": db_status,
        "timestamp": datetime.now().isoformat(),
        "careers_count": len(CAREER_PATHS),
        "options_count": len(QUESTION_TREE["options"])
    }

@app.get("/question")
async def get_first_question():
    """Get first question"""
    return {
        "id": QUESTION_TREE["id"],
        "question": QUESTION_TREE["question"],
        "level": QUESTION_TREE["level"],
        "options": list(QUESTION_TREE["options"].keys())
    }

@app.post("/answer")
async def process_answer(answer: AnswerRequest):
    """Process answer and return next question or career prediction"""
    try:
        print(f"\n📥 Received request:")
        print(f"   User: {answer.user_name}")
        print(f"   Path: {answer.current_path}")
        print(f"   Question ID: {answer.question_id}")
        
        # Handle empty path - return first question
        if not answer.current_path:
            return {
                "status": "continue",
                "question": {
                    "id": QUESTION_TREE["id"],
                    "question": QUESTION_TREE["question"],
                    "level": QUESTION_TREE["level"],
                    "options": list(QUESTION_TREE["options"].keys())
                }
            }
        
        # Navigate tree based on current path
        current = QUESTION_TREE
        
        for i, selection in enumerate(answer.current_path):
            # If current has "next", move to it first
            if "next" in current:
                current = current["next"]
            
            # Check if options exist
            if "options" not in current:
                return {
                    "status": "error", 
                    "message": "Invalid path - no more options available"
                }
            
            # Check if selection exists in options
            if selection not in current["options"]:
                return {
                    "status": "error",
                    "message": f"Selection '{selection}' not found in options"
                }
            
            current = current["options"][selection]
        
        # Check if we have next question
        if "next" in current:
            next_q = current["next"]
            print(f"➡️ Sending next question: {next_q['question']}")
            return {
                "status": "continue",
                "question": {
                    "id": next_q["id"],
                    "question": next_q["question"],
                    "level": next_q["level"],
                    "options": list(next_q["options"].keys())
                }
            }
        
        # If career_id exists, this is final answer
        elif "career_id" in current:
            career_id = current["career_id"]
            
            # Get features and make prediction
            features = torch.tensor([current["features"]], dtype=torch.float32)
            
            with torch.no_grad():
                output = model(features)
                confidence = torch.softmax(output, dim=1)[0][career_id].item() * 100
            
            confidence_rounded = round(confidence, 2)
            career = CAREER_PATHS[career_id]
            
            print(f"🎯 Career Prediction:")
            print(f"   Title: {career['title']}")
            print(f"   Confidence: {confidence_rounded}%")
            print(f"   User: {answer.user_name}")
            
            # Save to database
            db_success = save_user_career(
                user_name=answer.user_name,
                selections=answer.current_path,
                career_id=career_id,
                career_title=career["title"],
                confidence=confidence_rounded
            )
            
            if not db_success:
                print("⚠️ Warning: Data not saved to database")
            
            return {
                "status": "complete",
                "career": career,
                "confidence": confidence_rounded,
                "user_name": answer.user_name,
                "db_saved": db_success
            }
        
        else:
            return {
                "status": "error",
                "message": "Invalid path structure"
            }
            
    except Exception as e:
        print(f"❌ Server error in /answer: {e}")
        return {
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }

@app.post("/generate-pdf")
async def generate_career_pdf(data: FinalAnswer):
    """Generate PDF with career roadmap"""
    try:
        print(f"\n📄 Generating PDF for {data.user_name}")
        print(f"   Career ID: {data.career_id}")
        print(f"   Selections: {data.selections}")
        
        # Get career data
        if data.career_id not in CAREER_PATHS:
            raise HTTPException(status_code=404, detail="Career not found")
        
        career = CAREER_PATHS[data.career_id]
        
        # Create unique filename
        filename = f"career_roadmap_{data.user_name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.pdf"
        filepath = f"./pdfs/{filename}"
        
        # Ensure pdfs directory exists
        os.makedirs("./pdfs", exist_ok=True)
        
        # Create PDF
        c = canvas.Canvas(filepath, pagesize=letter)
        width, height = letter
        
        # Set margins
        margin = inch
        
        # Title
        c.setFont("Helvetica-Bold", 24)
        c.setFillColorRGB(0.1, 0.3, 0.6)  # Blue color
        c.drawString(margin, height - margin, f"Career Roadmap for {data.user_name}")
        
        # Career Title
        c.setFont("Helvetica-Bold", 20)
        c.setFillColorRGB(0.2, 0.5, 0.8)
        c.drawString(margin, height - margin - 40, career["title"])
        
        # Confidence
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0, 0.6, 0.2)  # Green color
        c.drawString(margin, height - margin - 70, f"Match Confidence: 85%")  # Placeholder
        
        # Line separator
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.line(margin, height - margin - 85, width - margin, height - margin - 85)
        
        # Description
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(margin, height - margin - 120, "📝 Description:")
        c.setFont("Helvetica", 10)
        text = c.beginText(margin, height - margin - 140)
        text.textLines(career["description"])
        c.drawText(text)
        
        # Skills
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, height - margin - 200, "🛠️ Required Skills:")
        c.setFont("Helvetica", 10)
        for i, skill in enumerate(career["skills"]):
            c.drawString(margin + 20, height - margin - 220 - (i * 15), f"• {skill}")
        
        # Roadmap
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, height - margin - 300, "🗺️ Learning Roadmap:")
        c.setFont("Helvetica", 10)
        for i, step in enumerate(career["roadmap"]):
            c.drawString(margin + 20, height - margin - 320 - (i * 20), step)
        
        # Salary
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0, 0.5, 0)
        c.drawString(margin, height - margin - 420, "💰 Expected Salary:")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin + 20, height - margin - 440, career["salary"])
        
        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(margin, margin - 10, "Generated by CareerMind AI | Best of luck for your journey! 🚀")
        c.drawString(margin, margin - 25, f"Generated on: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        c.save()
        
        print(f"✅ PDF generated: {filename}")
        
        return {
            "success": True,
            "download_url": f"/download-pdf/{filename}",
            "filename": filename,
            "message": "PDF generated successfully"
        }
        
    except Exception as e:
        print(f"❌ PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.get("/download-pdf/{filename}")
async def download_pdf(filename: str):
    """Download generated PDF"""
    filepath = f"./pdfs/{filename}"
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(
        filepath,
        media_type='application/pdf',
        filename=filename,
        headers={
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
    )

@app.get("/stats")
async def get_stats():
    """Get statistics about the system"""
    conn = get_db_connection()
    total_users = 0
    recent_users = []
    
    if conn:
        try:
            cursor = conn.cursor()
            
            # Count total records
            cursor.execute("SELECT COUNT(*) FROM user_careers")
            total_users = cursor.fetchone()[0]
            
            # Get recent users
            cursor.execute("""
                SELECT user_name, career_title, confidence, created_at 
                FROM user_careers 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            
            rows = cursor.fetchall()
            recent_users = [
                {
                    "user": row[0],
                    "career": row[1],
                    "confidence": row[2],
                    "created_at": row[3].isoformat() if row[3] else None
                }
                for row in rows
            ]
            
            cursor.close()
            conn.close()
            
        except Error as e:
            print(f"Stats error: {e}")
    
    return {
        "total_users": total_users,
        "recent_users": recent_users,
        "total_careers": len(CAREER_PATHS),
        "total_options": len(QUESTION_TREE["options"]),
        "server_time": datetime.now().isoformat()
    }

# ========================================
# 🎯 MAIN ENTRY POINT
# ========================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Starting CareerMind AI Backend...")
    print("="*50)
    setup_database()
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        reload=True  # Auto-reload on code changes
    )