#!/usr/bin/env python3
"""Entry point: ``python generator.py --help``.

    python generator.py --define   --schema job_postings.json
    python generator.py --generate --schema examples/job_postings.json --count 1000 --format csv
    python generator.py --validate --data output/job_postings/job_postings.csv --schema examples/job_postings.json
    python generator.py --export   --format parquet --output data.parquet
"""

from src.cli import main

if __name__ == "__main__":
    main()
