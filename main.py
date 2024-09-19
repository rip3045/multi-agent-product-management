# main.py

from team_agents import (
    SoftwareArchitectAgent,
    CoderAgent,
    CodeReviewerAgent,
    ProductManagerAgent,
    llm,
)



def main():
    # Instantiate agents
    architect = SoftwareArchitectAgent(llm_instance=llm)
    coder = CoderAgent(llm_instance=llm)
    reviewer = CodeReviewerAgent(llm_instance=llm)
    manager = ProductManagerAgent()

    # Step 1: Architect designs modules and descriptions
    product_description = "A calculator that can perform basic arithmetic operations."
    modules = architect.design_modules(product_description)
    print("Modules Designed:", modules)

    # Step 2: Coder implements modules iteratively
    all_code = ""
    for module in modules:
        module_name = module['name']
        module_description = module['description']
        code = coder.code_module(module_name, module_description)
        all_code += code + "\n"
        print(f"Code for module {module_name}:\n{code}\n")

    # Step 3: Reviewer checks and documents code iteratively
    for module in modules:
        module_name = module['name']
        code_review = reviewer.review_code(module_name)
        documentation = reviewer.document_code(module_name)
        print(f"Review for module {module_name}:\n{code_review}")
        print(f"Documentation for module {module_name}:\n{documentation}")

    # Step 4: Product manager runs and reviews the code
    success, output = manager.run_and_review(all_code)
    if success:
        print(f"Code executed successfully. Output:\n{output}")
    else:
        print(f"Code execution failed. Error:\n{output}")

if __name__ == "__main__":
    main()
