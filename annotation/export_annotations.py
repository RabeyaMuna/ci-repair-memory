#!/usr/bin/env python3
"""
Export annotations from browser localStorage to CSV file
This script reads the localStorage JSON and updates the CSV file on disk
Works for any annotator (Rabeya, Nakhla, etc.)
"""

import json
import csv
import sys
import os

def update_csv_with_annotations():
    """Update the CSV file with annotations from localStorage"""

    print("=" * 60)
    print("Export Annotations from localStorage to CSV")
    print("=" * 60)
    print()

    # Ask for annotator name
    print("Enter annotator name (e.g., rabeya, nakhla): ", end='')
    annotator = input().strip().lower()

    csv_filename = f'{annotator}_annotations.csv'

    # Check if CSV exists in current directory
    if not os.path.exists(csv_filename):
        print(f"\n✗ CSV file not found: {csv_filename}")
        print(f"Make sure you're running this script from the annotation folder")
        sys.exit(1)

    print()
    print("INSTRUCTIONS:")
    print("1. Open your annotation HTML file in browser")
    print("2. Open Browser Console (F12 or Cmd+Option+I)")
    print(f"3. Type: JSON.stringify(localStorage.getItem('annotations_{annotator}'))")
    print("4. Copy the output (the entire JSON string)")
    print("5. Paste it when prompted below")
    print()
    print("-" * 60)
    print()

    # Get localStorage data from user
    print("Paste the localStorage JSON here (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "" and len(lines) > 0:
            break
        lines.append(line)

    localStorage_json = "".join(lines)

    try:
        # Parse the localStorage data
        if localStorage_json.startswith('"') and localStorage_json.endswith('"'):
            localStorage_json = json.loads(localStorage_json)

        annotations = json.loads(localStorage_json)
        print(f"\n✓ Found {len(annotations)} annotated instances")

    except json.JSONDecodeError as e:
        print(f"\n✗ Error parsing JSON: {e}")
        print("Make sure you copied the complete output from the browser console")
        sys.exit(1)

    # Read the existing CSV
    try:
        with open(csv_filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"✓ Read {len(rows)} rows from CSV")

    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        sys.exit(1)

    # Update rows with annotations
    updated_count = 0
    for row in rows:
        instance_id = row['ID']
        if instance_id in annotations:
            ann = annotations[instance_id]
            row['Human_Failure_Types'] = ', '.join(ann.get('types', []))
            row['Human_Sub_Types'] = ', '.join(ann.get('subTypes', []))
            row['Human_Notes'] = ann.get('notes', '')
            updated_count += 1

    print(f"✓ Updated {updated_count} rows with annotations")

    # Write back to CSV
    with open(csv_filename, 'w', encoding='utf-8', newline='') as f:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Successfully wrote to: {csv_filename}")
    print()
    print("=" * 60)
    print("✓ DONE! Your annotations have been saved to the CSV file.")
    print("=" * 60)

if __name__ == '__main__':
    update_csv_with_annotations()
