# Product Requirements Document (PRD)
## Group 1 - AI Recycling
*Jack Stevenson, Nigel Higgs, Dillon Baldwin, Ramish Mohammad*
*11/27/2023*

# Problem Description
Standard recycling equipment is aging and requires extensive manpower to operate. Current versions of automated recycling systems lack flexibility in how they can be updated to better sort new items. It can be difficult to examine them analytically and determine what is being recycled incorrectly. In the problem that we are tasked with addressing, we have a set of recycling machines at plants in Portland. Each machine is able to pick out individual items from a conveyor belt using a camera and mechanical actuators. Our task is to improve the software running the machines so that it is more efficient and easier for operators to work with. We need to work on the machine learning algorithms behind the object identification process and ensure that new objects are recognized and classified correctly.

# Scope
The scope for this product will be limited to improving the current implementation of automated recycling software running in recycling centers in Portland. The goals include making it easier to recognize and add a new item to the computer vision model, making use of extensive data, allowing for fewer humans to be involved, automating more of the machine learning pipeline, increasing processing power and accuracy, and making the systems easier to use from a human perspective. We are limiting our scope to the systems in Portland because they are standardized and offer a well-defined problem.

From a software perspective, the scope involves a cloud-based machine learning pipeline that is able to take new imagery from the recycling plant to improve recognition of objects. It also involves a feature engineering approach to determine which real or derived features in the data are most relevant to object detection and classification. It needs to incorporate an API to connect the predictive backend to the recycling machine and databases to store images and feature sets.

# Use Cases
- A bottling company wants to know what proportion of its bottled drinks are recycled in Portland. We test this by collecting the analytical data from our system and comparing it to sales data.
- A recycling manager wants to separate PET plastic bottles from other plastics. We test this by defining a new class made up of several identification classes corresponding to different types of PET bottles.
- The recycling company wants to improve its accuracy. We test this by generating additional training data with artificially generated imagery, then train the model on the new data and test it on real examples.
- The plant operator wants to decrease the number of misclassified bottles. We can test this by comparing a sample of bottles classified using our approach to a sample from the previous approach.

# Purpose and Vision (Background)
The purpose and vision of this project are centered on promoting sustainability and fostering a circular economy by advancing the recycling sorting process with innovative computer vision and robotics technology. This initiative is designed to enhance the precision in identifying and separating various post-consumer waste materials to improve the overall quality and efficiency of recycling. The primary objective is to develop and implement a solution for select recycling plants in Portland that leverages deep learning techniques, such as YOLO, to automate the sorting process. This automation will facilitate continuous operation, significantly increasing operational efficiency across facilities. Additionally, the project aims to create new opportunities for revenue by supplying data to companies striving to enhance the eco-friendliness of their products, ensuring consistent improvements in recycling practices nationwide. We want to reduce the amount of human interaction necessary for plant operation and make it as easy as possible for new objects to be automatically recognized as novel classes.

# Stakeholders
- Recycling Centers
- CEO/CTO/Founders
- Engineering Managers
- Product Managers
- Engineering Team
- Marketing Team
- Legal Team
- Sales
- Users
- Companies of Recyclable Goods
- General Public

# Preliminary Context
## Assumptions
- Continuous Video Streaming: The project has access to a camera that streams continuously, capturing material movement on a conveyor belt at the recycling facility. This constant data stream is vital for real-time processing and analysis.
- State-of-the-Art Vision Models: It is assumed that advanced vision models, like YOLOv9, are suitable as the project's foundation. These models are known for their object labeling accuracy, which is essential for the initial identification of different materials. The project expects these models to be adaptable and trainable for specific recycling needs.
- Azure Integration: The project heavily relies on Azure's capabilities, assuming that Azure provides pre-built versions of these advanced vision models. This integration is anticipated to speed up the initial development phase, leveraging Azure's cloud computing resources and potential machine learning tools.
- Existing Software Compatibility: There is an expectation that the project can utilize existing software to interface with the robotic sorting equipment already in place. This implies a level of compatibility and integration between new and existing systems.
- Object Identification Capabilities: The team is confident in its ability to identify different types of bottles and materials based on distinguishable features like color and shape. This skill is crucial for accurate sorting and recycling.
- Collaboration with Recycling Technicians: The project plans to work closely with current recycling technicians. Their insights and concerns about the existing processes will guide improvements and ensure that the new system addresses real-world challenges effectively.
- Standardization of Machines and Input: There is an assumption that the recycling machines and the materials they process are standardized enough to allow the use of a uniform model across different machines. This uniformity would simplify the development process and make the model more universally applicable within the facility.

# Constraints
Our project, developed by a small team of relatively new computer science students, faces several challenges and constraints in developing an effective recycling sorting system. Our primary goal is ensuring that each recyclable object is accurately categorized, without being missed or missorted. To achieve this, our machine learning algorithms need to be capable of learning and properly sorting new objects. Additionally, we are operating under the constraint of a limited budget, which is a common challenge in the low-margin recycling business. This financial limitation requires us to build upon and enhance the current architecture of the machine learning model and pipeline, rather

## Requirements
### User Stories and Features (Functional Requirements)
| User Story | Feature | Priority | GitHub Issue | Dependencies |
|------------|---------|----------|--------------|--------------|
| As a machine operator, I want to see real-time information about sorting and be able to add new objects to the model. | TBD | Should Have | TBD | N/A |
| As a local bottling plant manager, I want to know what percentage of my bottles end up in the recycling. | TBD | Should Have | TBD | N/A |
| As the owner of the recycling center, I want the ML pipeline to run without human interaction. | TBD | Must Have | TBD | N/A |
| As a floor manager at the recycling center, I want to see analytics on how much recycling we are processing and how accurate our process is. | TBD | Should Have | TBD | N/A |

### Non-Functional Requirements
- New objects should be incorporated into the model within a day of them first being seen.
- The system should have a redundant design such that it will function if a single computer loses power or functionality.
- The system should be able to function for at least several hours if connectivity to the internet is lost.
- The system should be at least 10% more performant than the original system.
- The system should be able to classify an object in less than 5 seconds.
- The system should be able to classify 15 objects at the same time.
- The system should reboot itself if it loses power.
- Code should be documented and written with best practices in mind.

### Data Requirements
- PNG images of objects on the conveyor belt: These will be stored in a non-relational database for defining object classes and training the model.
- Counts of different objects: The number of times each object class is seen and sorted should be stored for analytical purposes.
- Errors: Objects that cannot be sorted correctly should be noted for further analysis and system refinement.
- Generated examples: If generative training is used, these would be stored similarly to the original PNG images.
- 3D models: If 3D models are used to model object classes, they will be stored in a non-relational database.

### Integration Requirements
- Microsoft Azure: Our ML pipeline will be run on Azure with most of the automated features being configured online.
- Robot control API: The system must be interoperable with the sorting machine to give it instructions.
- Database API: Standard for storing images and logging errors or analytics information.

### User Interaction and Design
The main user interface should be a basic dashboard that informs about the status of the machine without getting into too much detail about the technology running it. An operator should be able to troubleshoot and identify issues with the machine or computer without looking at or modifying the ML pipeline. This is a very basic idea of what an operator on the recycling center floor might see. The log could include errors in the machine like jams or mechanical failures if that data is available from the machine.

## Milestones and Timeline
- Nov 14: Finish becoming familiar with the current implementation of automated recycling
- Dec 14: Prototype new data collection and preprocessing methods. Iterate through ideas and mock up in Azure as appropriate
- Jan 14: Have a running prototype dashboard for the user side of the system
- Jan 28: Receive live information from the current model to feed into the dashboard
- Feb 14: Deploy new ML pipeline with improvements in data collection and automation
- Feb 28: Sandbox test pipeline with existing data and find limitations
- Mar 28: Finish functional ML pipeline with meaningful improvements
- Apr 1: Deploy new ML pipeline on a single sorting machine to test in real life
- Apr 14: Finish debugging initial hardware and software setup
- Apr 28: Fine-tune model with information from real-life implementation
- May 7: Finalize automation protocols
- May 14: Deploy across other machines
- May 21: Observe machines in action and fix any issues

## Goals and Success Metrics
| Goal | Metric | Baseline | Target | Tracking Method |
|------|--------|----------|--------|------------------|
| Decrease identification time | Average time to recognition | 5 sec | 2 sec | Analytics |
| Decrease necessary manpower | Man-hours per hour of system operation | 0.5 hrs | <0.1 hrs | Interview with Manager |
| Improve accuracy | Percent of items sorted | 80% | 92% | Analytics |
| Increase number of sorting categories | Number of categories | 7 | 15 | Analytics |

## Open Questions
- What do the actual machines look like on the sorting floor? Are there periods when they are inactive, or do they run nonstop?
- What do current manpower requirements look like at the recycling center?
- Are there any differences between the machines at different recycling centers? If so, how much will this affect our project?
- What categories of recycling are the most important to keep pure? Are certain types of plastics more easily recycled than others? If so, where should our system err in a case of unsure classification?
- If we want to track the brands of recycled items, what are the manufacturers of interest? Are we selling them data or giving it for free?

## Out of Scope
- Flexible system - Our understanding is that we are iterating on an existing system for a set number of recycling machines, so we are planning to cater only to those machines.
- Business case - We are not responsible for justifying the business case for this system. We will strive to keep it reasonable in cost, but we won’t be responsible for making business decisions on behalf of the operator.
- Advanced security - We will adhere to best practices, but we understand that our system functions in a low-risk environment. We are not focused on potential attacks from cybercriminals.
- Long-term maintenance - We will design a system that works well, but we are not responsible for patching issues or implementing new features after we complete the project at the end of this year.
- Mechanical and hardware setup - It is not our responsibility to configure the sorting machine or its connection to a controlling computer. We are using what already exists.
