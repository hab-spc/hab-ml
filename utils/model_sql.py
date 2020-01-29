# Standard dist imports
import sqlite3
import logging

#project level imports
from utils.config import opt, set_config
from utils.constants import *

class model_sql:
    """
    Query and store models from sql lite database
    """
    
    def __init__(self):
        """
        Open connection to the sqlite model database
        """
        self.conn = sqlite3.connect('/data6/plankton_test_db_new/model/model.db')
        self.c = self.conn.cursor()
        self.logger = logging.getLogger('Model_SQL')

    def sqlite_command(self, command):
        """
        Helper Function. Only used within this class. Used to execute sqlite commands.
        """
        stop = False
        err = 0
        try:
            self.c.execute(command)
        except sqlite3.Error as e:
            self.logger.debug(e)
            err = 1
        except Exception as e:
            self.logger.debug(e)
            err = 1
        except Error as e:
            self.logger.debug(e)
            err = 1
        finally:
            if err == 0:
                stop = True

        return [self.c.fetchall(), stop]

    def find_model_id(self):
        """
        Helper Function. Used within this class. Used to get user inputs to query the model, and get the model id.
        """
        stop = False
        while (stop == False):
            command = input('Enter query command (ex. SELECT * FROM models WHERE train_acc > 90) : \n')
            [rows, stop] = self.sqlite_command(command)      
        for row in rows:
            self.logger.info(row)

        stop = False
        while (stop == False):
            command = input('Enter the ID number of the model: \n')
            row_id = int(command)
            [rows, stop] = self.sqlite_command("SELECT * FROM models WHERE ID = " + command)      
        self.logger.info('The model you find is: ')
        self.logger.info(rows)
        return row_id
    
    def find_model_dir_path(self):
        """
        Used in Trainer in resume or VAL mode to get the model_dir
        """
        rowid = str(self.find_model_id())
        self.c.execute("SELECT model_dir_path FROM models WHERE ID = " + rowid)
        return self.c.fetchall()[0][0]
    
    def save_model(self):
        """
        Save model to sqlite model database. Used at the end of main.py
        """
        opt.add_comm = input('Any Addtional Comment to this model? \n')
        with self.conn:
            self.c.execute(
                "INSERT INTO models VALUES (NULL, :type, :m_d_p, :f_l, :f_t, :lr, :bs, :train_acc, :test_acc, :ep, :date, :cl, :comment)",
                {'type': opt.arch, 'm_d_p': opt.model_dir, 'f_l': opt.freezed_layers, 'f_t': 'n', 'lr': opt.lr,
                 'bs': opt.batch_size, 'train_acc': opt.train_acc, 'test_acc': opt.test_acc, 'ep': opt.epochs,
                 'date': opt.model_date, 'cl': opt.class_num, 'comment': opt.add_comm})

    def close(self):
        """
        Close the sqlite connection
        """
        self.conn.close()
