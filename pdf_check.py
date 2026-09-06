import sys, pdfplumber

with pdfplumber.open(sys.argv[1]) as pdf:
    blank = []
    for i, page in enumerate(pdf.pages, 1):
        text = page.extract_text() or ""
        tables = page.extract_tables()
        if not text.strip():
            blank.append(i)
        for t in tables:
            print(f"p{i}: {t[0]}")
    print(f"\ntotal pages: {len(pdf.pages)}")
    print(f"blank pages: {blank}")