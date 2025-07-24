def similarity_reasoning_prompt(user_skill: str, job_requirement: str) -> str:
    """
    Generates a prompt for reasoning about the similarity between a CV line and a job requirement line.

    Args:
        user_skill (str): A line from the user's CV.
        job_requirement (str): A line from the job description.

    Returns:
        str: The formatted prompt for similarity reasoning.
    """

    system_role = "You are an expert in analyzing CVs and job descriptions. Your task is to determine the similarity between a line from a CV and a line from a job description"
    variabale_assignment = f"""

        This is an user skill line from CV: {user_skill}
        This is a job requirement line: {job_requirement}
    """
    steps_and_outputs = """

        # Steps

        - Carefully read the CV line and requirement line.
        - Identify and list explicit or implied variables and tools present in both.
        - Analyze how these elements (variables, tools, skills, experiences) overlap or align, and why this demonstrates meaningful similarity.
        - Summarize your reasoning clearly before writing your final conclusion.
        - Provide 2-3 example pairs (using [placeholder text] for complex variables/tools if necessary) that directly address variables and tools alignment.

        # Output Format

        - Begin with a "Reasoning" section with detailed analysis.
        - Follow with a short "Conclusion" summarizing their similarity based on variables/tools.
        - List 2-3 clear example pairs (CV line + Requirement line), using placeholders as needed for variables/tools.

        # Examples

        Reasoning:
        Both lines name the variable "customer churn rate" and the tool "SQL." The CV line shows practical use of SQL to analyze churn, and the requirement expects experience with the same variable and tool. Their alignment is in applying SQL for data analysis tasks focused on customer behavior metrics.

        Conclusion:
        These lines are similar because they both require using SQL for analyzing the customer churn rate.

        Example Pairs:
        CV line: "Built logistic regression models in R to predict sales volume."
        Requirement: "Experience with R and regression techniques to model KPIs."

        CV line: "Utilized Tableau dashboards to visualize A/B test results."
        Requirement: "Proficiency with Tableau for presenting experimentation outcomes."

        CV line: "Automated inventory updates with Python and REST APIs."
        Requirement: "Ability to use Python and APIs for process automation."

        (Reminder: Always reason thoroughly before concluding, with keen focus on variables and tools. Use illustrative examples.)

        # Notes

        - Emphasize alignment based on use or knowledge of variables (specific data types, fields, measurable quantities) and tools (software, programming languages, analytic methods).
        - Examples should illustrate explicit alignment around variables/tools, not just general skills.
        - Maintain clear structured reasoning before any conclusion.
        - Use [placeholder text] to represent particularly long/complex variable or tool descriptions where necessary.
    """

    return f"{system_role}\n{variabale_assignment}\n{steps_and_outputs}"

def job_requirement_extracting_prompt(job_description: str) -> str:
    """
    Generates a prompt for extracting job requirements from a job description.

    Args:
        job_description (str): The job description text.

    Returns:
        str: The formatted prompt for extracting job requirements.
    """
    
    system_role = "You are an expert in analyzing job descriptions. Your task is to extract structured information about the job requirements."
    variabale_assignment = f"""
        This is the job description: {job_description}
    """
    steps_and_outputs = """
        # Steps
        - Carefully read the job description.
        - Identify and list all explicit and implied requirements, including skills, experiences, tools, and qualifications.
        - Structure the extracted information in a clear format.

        # Output Format
        - Provide a structured list of requirements with clear headings for each category (e.g., Skills, Experience, Tools).
        - Use bullet points for clarity.

        # Example Output
        Skills:
        - Python programming
        - Data analysis with Pandas
        - Machine learning algorithms

        Experience:
        - 3+ years in software development
        - Experience with cloud platforms (AWS, Azure)

        Tools:
        - Git version control
        - Docker containerization

        (Reminder: Focus on extracting detailed and specific requirements that are essential for the role.)
    """

    return f"{system_role}\n{variabale_assignment}\n{steps_and_outputs}"


