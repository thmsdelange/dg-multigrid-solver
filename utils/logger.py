import coloredlogs, logging, logging.handlers, sys

formatter = logging.Formatter('%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s')
coloredformatter = coloredlogs.ColoredFormatter('%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s') 

class Logger:
    def __init__(self, namespace, settings):
        ### create a logger object
        self.logger = logging.getLogger(namespace)
        self.logger.setLevel(getattr(logging, settings.logging.loglevel))

        ### create consolehandler
        self.consolehandler = logging.StreamHandler(sys.stdout)
        self.consolehandler.setLevel(getattr(logging, settings.logging.loglevel))
        self.consolehandler.setFormatter(coloredformatter)
        
        ### create a filehandler object
        self.filehandler = logging.handlers.RotatingFileHandler(filename='logs/debug.log', maxBytes=1024000, backupCount=10, mode="a")
        self.filehandler.setLevel(getattr(logging, settings.logging.loglevel))
        self.filehandler.setFormatter(formatter)
        
        # self.filehandler.setFormatter(formatter)
        self.logger.addHandler(self.consolehandler)
        if settings.logging.write_to_file: self.logger.addHandler(self.filehandler)

        ### Install the coloredlogs module on the root logger
        coloredlogs.install(level=settings.logging.loglevel, logger=self.logger)