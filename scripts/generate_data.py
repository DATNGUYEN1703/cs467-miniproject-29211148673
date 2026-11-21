"""Create the original, fictional dataset. No real student records are used.

Run from any directory: python scripts/generate_data.py
Seed and generation assumptions are fixed for reproducibility.
"""

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026

# code, title, department, syllabus, skill tags. All syllabi are original.
COURSES = [
    ("SWE301", "Backend API Development", "Software Engineering", "Build backend APIs with Python and FastAPI. Design REST endpoints, authentication, validation, database integration and automated API tests.", "backend APIs; Python; FastAPI; REST; databases"),
    ("SWE302", "Cloud Application Deployment", "Software Engineering", "Deploy backend applications to cloud platforms. Practice Docker containers, cloud deployment, CI CD pipelines, monitoring and scalable API hosting.", "cloud deployment; Docker; backend; APIs; CI CD"),
    ("SWE303", "Database Systems and SQL", "Software Engineering", "Design relational databases for backend services. Write SQL joins, optimize queries, create indexes and manage transactions in PostgreSQL.", "databases; SQL; PostgreSQL; backend; transactions"),
    ("SWE304", "Distributed Microservices", "Software Engineering", "Develop distributed backend microservices using message queues and REST APIs. Study service discovery, fault tolerance and cloud deployment patterns.", "microservices; backend APIs; cloud deployment; message queues"),
    ("SWE305", "DevOps Engineering", "Software Engineering", "Automate software delivery with Git workflows, container builds and continuous integration. Deploy services with Kubernetes and collect observability metrics.", "DevOps; Kubernetes; Docker; CI CD; observability"),
    ("SWE306", "NoSQL and Data Storage", "Software Engineering", "Compare document databases, key value stores and relational databases. Build backend data services with MongoDB, Redis caching and consistency tradeoffs.", "databases; NoSQL; MongoDB; Redis; backend"),
    ("AI301", "Applied Machine Learning", "Artificial Intelligence", "Train machine learning models for classification and regression using Python. Evaluate predictive models with cross validation and feature engineering on tabular data.", "machine learning; Python; classification; regression; data"),
    ("AI302", "Deep Learning Foundations", "Artificial Intelligence", "Build neural networks for prediction with tensors, backpropagation and gradient descent. Train deep learning models and examine regularization and model evaluation.", "deep learning; neural networks; Python; prediction"),
    ("AI303", "Natural Language Processing", "Artificial Intelligence", "Analyze text data with tokenization, TF IDF, word embeddings and text classification. Build natural language processing applications for search and sentiment analysis.", "natural language processing; text; machine learning; TF IDF"),
    ("AI304", "Data Mining and Analytics", "Artificial Intelligence", "Discover patterns in data using clustering, association rules and anomaly detection. Prepare datasets and evaluate data mining models in Python.", "data mining; analytics; clustering; Python; machine learning"),
    ("AI305", "Computer Vision", "Artificial Intelligence", "Process image data and train convolutional neural networks for image classification. Study visual features, object detection and deep learning evaluation.", "computer vision; deep learning; neural networks; image classification"),
    ("AI306", "Recommendation Systems", "Artificial Intelligence", "Build personalized recommendation systems with content based models, collaborative filtering and hybrid ranking. Evaluate user item data with precision and recall.", "recommendation systems; machine learning; collaborative filtering; data"),
    ("SEC301", "Network Security", "Cybersecurity", "Secure network protocols, configure firewalls and inspect traffic. Investigate intrusion detection, network attacks and practical defense strategies.", "network security; firewalls; intrusion detection; defense"),
    ("SEC302", "Ethical Hacking", "Cybersecurity", "Perform authorized penetration testing in isolated labs. Identify web vulnerabilities, document security findings and recommend remediation for vulnerable systems.", "security; penetration testing; vulnerabilities; ethical hacking"),
    ("SEC303", "Applied Cryptography", "Cybersecurity", "Use symmetric encryption, public key cryptography, digital signatures and secure key management. Analyze protocol security and common implementation mistakes.", "cryptography; encryption; signatures; security"),
    ("SEC304", "Secure Software Design", "Cybersecurity", "Apply threat modeling and secure coding to web software. Prevent injection attacks, protect authentication flows and review API security controls.", "security; secure coding; authentication; API security"),
    ("SEC305", "Digital Forensics", "Cybersecurity", "Collect and preserve digital evidence from disks and logs. Reconstruct security incidents with forensic tools and write reproducible investigation reports.", "digital forensics; incident response; security; logs"),
    ("SEC306", "Cloud Security", "Cybersecurity", "Protect cloud infrastructure through identity management, least privilege and encryption. Audit cloud configurations and monitor security incidents in container services.", "cloud security; identity; encryption; containers; security"),
    ("UX301", "Human Computer Interaction", "Design", "Study human centered design, usability and accessibility. Interview users, create interaction prototypes and evaluate interfaces with usability testing.", "UX; usability; interaction design; accessibility"),
    ("UX302", "User Interface Prototyping", "Design", "Create interface wireframes and interactive prototypes using design systems. Practice visual hierarchy, typography and usability reviews for mobile applications.", "UI; prototypes; design systems; visual design"),
    ("UX303", "Frontend Web Development", "Design", "Build accessible web interfaces with HTML, CSS and JavaScript. Implement responsive layouts, component state and frontend API integration.", "frontend; JavaScript; HTML; CSS; accessibility"),
    ("UX304", "Mobile Interaction Design", "Design", "Design mobile user journeys and touch interactions. Prototype navigation flows and test mobile interfaces for usability and accessibility.", "mobile design; UX; prototypes; usability"),
    ("UX305", "Game Design Studio", "Design", "Prototype interactive games with player feedback loops. Study game mechanics, narrative design, level balancing and playtesting methods.", "game design; prototyping; interaction; playtesting"),
    ("UX306", "Information Visualization", "Design", "Communicate data with charts and interactive visual interfaces. Choose visual encodings, critique misleading graphics and conduct usability testing of dashboards.", "data visualization; dashboards; design; usability"),
    ("BUS301", "Business Intelligence", "Business Analytics", "Transform business data into decision dashboards. Define key performance indicators, model warehouse data and communicate analytics findings to managers.", "business intelligence; analytics; dashboards; data warehouse"),
    ("BUS302", "Digital Marketing Analytics", "Business Analytics", "Analyze marketing campaigns with segmentation and conversion metrics. Evaluate experiments and present actionable business insights from customer data.", "marketing analytics; segmentation; experiments; business"),
    ("BUS303", "Technology Entrepreneurship", "Business Analytics", "Explore technology venture creation, customer discovery and business models. Validate product assumptions and pitch an evidence based startup proposal.", "entrepreneurship; business models; product; customer discovery"),
    ("BUS304", "Product Management", "Business Analytics", "Prioritize product roadmaps using customer research and business metrics. Write product requirements, define experiments and coordinate software delivery decisions.", "product management; business; experiments; roadmaps"),
    ("BUS305", "Financial Data Analytics", "Business Analytics", "Analyze financial time series and portfolio data with Python. Build descriptive risk reports, inspect forecast errors and communicate financial analytics responsibly.", "financial analytics; Python; business; time series"),
    ("BUS306", "Operations Analytics", "Business Analytics", "Use optimization and simulation for business operations. Model scheduling, inventory and resource allocation with data driven decision techniques.", "operations; analytics; optimization; simulation; business"),
    ("CS301", "Algorithms for Problem Solving", "Computer Science", "Design efficient graph and dynamic programming algorithms. Analyze time complexity and apply algorithmic techniques to practical computing problems.", "algorithms; graphs; dynamic programming; complexity"),
    ("CS302", "Statistical Computing", "Computer Science", "Use Python to estimate distributions, confidence intervals and hypothesis tests. Apply statistical reasoning to data analysis and reproducible experiments.", "statistics; Python; data analysis; experiments"),
    ("CS303", "Internet of Things", "Computer Science", "Connect embedded sensors to network services. Collect sensor data with MQTT, build device prototypes and examine reliability of connected systems.", "IoT; sensors; MQTT; embedded systems"),
    ("CS304", "Software Testing", "Computer Science", "Design unit, integration and property based tests. Measure coverage, automate regression testing and diagnose failures in reliable software systems.", "software testing; automation; quality; Python"),
    ("CS305", "Responsible Artificial Intelligence", "Computer Science", "Examine fairness, transparency and privacy in machine learning. Audit data and model behavior, document limitations and discuss responsible AI deployment.", "responsible AI; machine learning; fairness; data; privacy"),
    ("SWE390", "Serverless Cloud APIs (New Elective)", "Software Engineering", "Build serverless backend APIs with cloud functions and managed databases. Practice cloud deployment, event driven API design, database integration and monitoring without managing servers.", "serverless; backend APIs; cloud deployment; databases; cloud functions"),
]


def generate():
    rng = random.Random(SEED)
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    items = [dict(zip(
        ["item_id", "course_code", "title", "department", "syllabus_text", "skills_tags"],
        [f"C{i:03d}", *row],
    )) for i, row in enumerate(COURSES, 1)]
    with (data_dir / "items.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)

    comments = {
        1: ["The workload did not suit my interests.", "I prefer a different career direction.", "The subject was not a good fit for me."],
        2: ["Some useful topics but limited interest.", "The labs were outside my main goals.", "I found the material difficult to connect with."],
        3: ["Useful introduction with a few challenging labs.", "A balanced elective for exploring the subject.", "Helpful concepts although I need more practice."],
        4: ["Practical projects helped build my skills.", "The material supports my career plans.", "Clear examples and useful hands on work."],
        5: ["Excellent fit for my preferred specialization.", "I would recommend the projects to classmates.", "Strong practical skills for my future career."],
    }
    ratings = []
    for student in range(1, 15):
        # U001 is the required AI/data enthusiast; U002 is backend oriented.
        preferred_group = 1 if student == 1 else (0 if student == 2 else (student - 3) % 5)
        preferred = list(range(preferred_group * 6 + 1, preferred_group * 6 + 7))
        others = [i for i in range(1, 36) if i not in preferred]
        selected = sorted(preferred + rng.sample(others, 12))
        for course in selected:
            group = (course - 1) // 6
            if student == 1 and course in preferred:
                rating = 5
            elif group == preferred_group:
                rating = rng.choices([3, 4, 5], weights=[1, 4, 6])[0]
            elif course in [31, 32, 34, 35]:
                rating = rng.choices([2, 3, 4, 5], weights=[1, 3, 4, 2])[0]
            else:
                rating = rng.choices([1, 2, 3, 4, 5], weights=[1, 3, 5, 2, 1])[0]
            ratings.append({"user_id": f"U{student:03d}", "item_id": f"C{course:03d}",
                            "rating": rating, "feedback_comment": rng.choice(comments[rating])})
    with (data_dir / "ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ratings[0]))
        writer.writeheader()
        writer.writerows(ratings)
    print(f"Created {len(items)} fictional courses and {len(ratings)} ratings from 14 simulated senior students.")
    print(f"Missing user-item entries: {1 - len(ratings) / (14 * len(items)):.1%}; C036 has no ratings.")


if __name__ == "__main__":
    generate()
