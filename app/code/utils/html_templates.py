"""
HTML templates for generating result pages
"""

def get_results_page_template(output_file_name):
    """
    Returns HTML template for results page with a link to the output file
    
    Args:
        output_file_name: Name of the output file to link to
        
    Returns:
        HTML content as a string
    """
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Combat DC Results</title>
</head>
<body>
    <h1>Results</h1>
    <p><a href="{output_file_name}">Download {output_file_name}</a></p>
</body>
</html>
"""