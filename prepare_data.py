import pdfplumber

# List your files here (PDFs and plain text)
files = [
    "Why Does He Do That.txt",
    "TX Family Code.txt",
    "qap-handbook.txt",
    "myths excuses and realities.txt",
    "Myths about abusers.txt",
    "mississippi.txt",
    "minnesota.txt",
    "michigan.txt",
    "mass.txt",
    "maryland.txt",
    "maine.txt",
    "louisiana.txt",
    "kentucky.txt",
    "kansas.txt",
    "iowa.txt",
    "indiana.txt",
    "illinois.txt",
    "idaho.txt",
    "The Enduring Impacts of Reunification Camps and Next Steps for Legislation.txt",
    "hawaii.txt",
    "georgia.txt",
    "new_sitemap.xml",  # We’ll handle this separately
    "ukpgaen_20200011_en.pdf",
    "GCP-DBA.txt",
    "florida.txt",
    "FamilyLaw1975.pdf",
    "family code - coparenting.txt",
    "Divorce_by_State_Handout.pdf",
    "divorce poison.txt",
    "delaware.txt",
    "dangers of couple counseling.txt",
    "da189862v1898n42167.pdf",
    "coparenting.txt",
    "connecticut.txt",
    "colorado.txt",
    "california.txt",
    "arkansas.txt",
    "arizona.txt",
    "Another Cancelled Visitation with Seven Minutes Notice.txt",
    "all-states.txt",
    "alabama.txt",
    "14 steps of change.txt"
]

# Process each file and write incrementally to training_data.txt
try:
    with open("training_data.txt", "w", encoding='utf-8') as outfile:
        # Process each file
        for file_name in files:
            if file_name.endswith(".pdf"):
                # Handle PDFs
                try:
                    with pdfplumber.open(file_name) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                outfile.write(page_text + "\n")
                except Exception as e:
                    print(f"Error with PDF {file_name}: {e}")
            elif file_name.endswith(".xml"):
                # Skip the sitemap for now
                print(f"Skipping sitemap {file_name} - will handle separately")
            else:
                # Handle plain text
                try:
                    with open(file_name, "r", encoding='utf-8') as file:
                        outfile.write(file.read() + "\n")
                except Exception as e:
                    print(f"Error with text file {file_name}: {e}")
except Exception as e:
    print(f"Error writing to training_data.txt: {e}")
    exit(1)

print("Text extracted and saved from all files!")