# Import db from flask_sqlalchemy directly to avoid circular imports
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from datetime import datetime

def backup_database_to_csv():
    """Backup database tables to CSV files"""
    try:
        # Export all tables
        tables = {
            'employees': 'SELECT * FROM employee',
            'skills': 'SELECT * FROM skill', 
            'employee_skills': 'SELECT * FROM employee_skill',
            'projects': 'SELECT * FROM project',
            'project_skills': 'SELECT * FROM project_skill',
            'development_paths': 'SELECT * FROM development_path',
            'matching_results': 'SELECT * FROM matching_result'
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for table_name, query in tables.items():
            df = pd.read_sql(query, db.engine)
            filename = f'backup_{table_name}_{timestamp}.csv'
            df.to_csv(filename, index=False)
            print(f"Backed up {table_name} to {filename}")

        print("Database backup completed successfully!")

    except Exception as e:
        print(f"Error backing up database: {e}")

def restore_database_from_csv(backup_timestamp):
    """Restore database from CSV backup files"""
    try:
        # Clear existing data
        db.drop_all()
        db.create_all()

        # Restore data
        restore_files = {
            'employees': f'backup_employees_{backup_timestamp}.csv',
            'skills': f'backup_skills_{backup_timestamp}.csv',
            'projects': f'backup_projects_{backup_timestamp}.csv',
            'development_paths': f'backup_development_paths_{backup_timestamp}.csv',
            'employee_skills': f'backup_employee_skills_{backup_timestamp}.csv',
            'project_skills': f'backup_project_skills_{backup_timestamp}.csv',
            'matching_results': f'backup_matching_results_{backup_timestamp}.csv'
        }

        for table_name, filename in restore_files.items():
            try:
                df = pd.read_csv(filename)
                df.to_sql(table_name, db.engine, if_exists='append', index=False)
                print(f"Restored {table_name} from {filename}")
            except FileNotFoundError:
                print(f"Backup file {filename} not found, skipping {table_name}")

        print("Database restore completed!")

    except Exception as e:
        print(f"Error restoring database: {e}")

def get_database_stats():
    """Get database statistics"""
    try:
        stats = {}

        # Count records in each table
        stats['employees'] = db.session.execute('SELECT COUNT(*) FROM employee').scalar()
        stats['skills'] = db.session.execute('SELECT COUNT(*) FROM skill').scalar()
        stats['projects'] = db.session.execute('SELECT COUNT(*) FROM project').scalar()
        stats['employee_skills'] = db.session.execute('SELECT COUNT(*) FROM employee_skill').scalar()
        stats['project_skills'] = db.session.execute('SELECT COUNT(*) FROM project_skill').scalar()
        stats['development_paths'] = db.session.execute('SELECT COUNT(*) FROM development_path').scalar()
        stats['matching_results'] = db.session.execute('SELECT COUNT(*) FROM matching_result').scalar()

        return stats

    except Exception as e:
        print(f"Error getting database stats: {e}")
        return {}
