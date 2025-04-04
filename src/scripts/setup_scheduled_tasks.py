import os
import sys
import platform
import subprocess
import logging
from datetime import datetime
import shutil

# add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ensure log directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"setup_scheduled_tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_windows_startup_script():
    """create .bat file for windows startup"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scraper_script = os.path.join(project_root, 'src', 'scripts', 'scraper.py')
    web_app_script = os.path.join(project_root, 'src', 'web', 'app.py')
    python_exe = sys.executable
    
    # create startup dir if it doesn't exist
    startup_dir = os.path.join(project_root, 'startup')
    os.makedirs(startup_dir, exist_ok=True)
    
    # create bat file for scraper
    scraper_bat_path = os.path.join(startup_dir, 'run_scraper.bat')
    with open(scraper_bat_path, 'w') as f:
        f.write(f'@echo off\n')
        f.write(f'echo Running news scraper...\n')
        f.write(f'"{python_exe}" "{scraper_script}"\n')
        f.write(f'echo Scraper completed. Starting web app...\n')
        f.write(f'"{python_exe}" "{web_app_script}"\n')
        f.write(f'echo Web app started.\n')
        f.write(f'pause\n')
    
    # create shortcut in windows startup folder
    try:
        # user startup folder
        user_startup = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup')
        
        # copy bat file or create shortcut
        if os.path.exists(user_startup):
            dest_path = os.path.join(user_startup, 'NewsAnalyzerStartup.bat')
            shutil.copy2(scraper_bat_path, dest_path)
            logger.info(f"startup script copied to: {dest_path}")
            return True
        else:
            logger.warning(f"windows startup folder not found: {user_startup}")
            logger.info(f"manually copy {scraper_bat_path} to your startup folder")
            return False
    except Exception as e:
        logger.error(f"error creating startup script: {str(e)}")
        logger.info(f"manually copy {scraper_bat_path} to your startup folder")
        return False

def setup_windows_task():
    """setup scheduled task on windows"""
    try:
        # get absolute paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        scraper_script = os.path.join(project_root, 'src', 'scripts', 'scraper.py')
        python_exe = sys.executable
        
        # create task command
        task_name = "NewsScraperDaily"
        command = f'schtasks /create /tn {task_name} /tr "\\"{python_exe}\\" \\"{scraper_script}\\"" /sc daily /st 08:00'
        
        # run command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("successfully created windows scheduled task")
            logger.info(f"task will run daily at 08:00, executing: {python_exe} {scraper_script}")
            
            # also set up startup script
            create_windows_startup_script()
            
            return True
        else:
            logger.error(f"failed to create windows scheduled task: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"error setting up windows task: {str(e)}")
        return False

def create_linux_startup_script():
    """create startup script for linux"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scraper_script = os.path.join(project_root, 'src', 'scripts', 'scraper.py')
    web_app_script = os.path.join(project_root, 'src', 'web', 'app.py')
    python_exe = sys.executable
    
    # create startup dir
    startup_dir = os.path.join(project_root, 'startup')
    os.makedirs(startup_dir, exist_ok=True)
    
    # create shell script
    startup_script_path = os.path.join(startup_dir, 'run_news_analyzer.sh')
    with open(startup_script_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# News Analyzer startup script\n\n')
        f.write(f'echo "Running news scraper..."\n')
        f.write(f'{python_exe} {scraper_script}\n\n')
        f.write(f'echo "Scraper completed. Starting web app..."\n')
        f.write(f'{python_exe} {web_app_script}\n')
    
    # make executable
    os.chmod(startup_script_path, 0o755)
    
    # create desktop entry
    desktop_entry_path = os.path.join(startup_dir, 'news-analyzer.desktop')
    with open(desktop_entry_path, 'w') as f:
        f.write('[Desktop Entry]\n')
        f.write('Type=Application\n')
        f.write('Name=News Analyzer\n')
        f.write('Comment=News analysis with AI\n')
        f.write(f'Exec={startup_script_path}\n')
        f.write('Terminal=true\n')
        f.write('X-GNOME-Autostart-enabled=true\n')
    
    # try to copy to autostart directory
    try:
        autostart_dir = os.path.expanduser('~/.config/autostart')
        if not os.path.exists(autostart_dir):
            os.makedirs(autostart_dir, exist_ok=True)
        
        dest_path = os.path.join(autostart_dir, 'news-analyzer.desktop')
        shutil.copy2(desktop_entry_path, dest_path)
        logger.info(f"startup entry created at: {dest_path}")
        return True
    except Exception as e:
        logger.error(f"error creating autostart entry: {str(e)}")
        logger.info(f"manually copy {desktop_entry_path} to ~/.config/autostart/ directory")
        return False

def setup_linux_cron():
    """setup cron job on linux/mac"""
    try:
        # get absolute paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        scraper_script = os.path.join(project_root, 'src', 'scripts', 'scraper.py')
        python_exe = sys.executable
        
        # create cron entry
        cron_entry = f"0 8 * * * {python_exe} {scraper_script} >> {log_dir}/scraper_cron_$(date +\\%Y\\%m\\%d).log 2>&1\n"
        
        # check existing crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        
        if result.returncode == 0:
            existing_crontab = result.stdout
            
            # check if entry already exists
            if cron_entry.strip() in existing_crontab:
                logger.info("cron job already exists")
                return True
                
            # add new entry
            new_crontab = existing_crontab + cron_entry
        else:
            # create new crontab
            new_crontab = cron_entry
        
        # write new crontab
        with open('/tmp/news_scraper_crontab', 'w') as f:
            f.write(new_crontab)
            
        # install new crontab
        result = subprocess.run(['crontab', '/tmp/news_scraper_crontab'], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("successfully created cron job")
            logger.info(f"job will run daily at 08:00, executing: {python_exe} {scraper_script}")
            
            # also set up startup script
            create_linux_startup_script()
            
            return True
        else:
            logger.error(f"failed to create cron job: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"error setting up cron job: {str(e)}")
        return False

def main():
    """main function for setting up scheduled tasks"""
    logger.info("setting up scheduled task and startup script for news analyzer")
    
    # determine operating system
    system = platform.system()
    
    if system == 'Windows':
        logger.info("detected windows operating system")
        setup_windows_task()
    elif system in ['Linux', 'Darwin']:  # Linux or Mac
        logger.info(f"detected {system} operating system")
        setup_linux_cron()
    else:
        logger.error(f"unsupported operating system: {system}")
        logger.info("please set up a scheduled task manually to run the scraper script daily")
    
    logger.info("setup completed")
    logger.info("\nInstructions for running at startup:")
    
    if system == 'Windows':
        logger.info("1. A startup script has been created in the 'startup' folder")
        logger.info("2. To run the application at system startup, use the shortcut in your Startup folder")
        logger.info("3. Or run manually the script: startup/run_scraper.bat")
    else:
        logger.info("1. A startup script has been created in the 'startup' folder")
        logger.info("2. To run the application at system startup, add the desktop entry to your autostart folder")
        logger.info("3. Or run manually the script: startup/run_news_analyzer.sh")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"unexpected error: {str(e)}")