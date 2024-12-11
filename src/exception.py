import sys, os
import logging

from src.logger import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
def error_message_details(error, error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno

    return f"The error has been occured in python scipt name: {file_name} , line number: {line_no} , error message: {error}"


class StudentException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error=error_message, error_details=error_details)

    def __str__(self):
        return self.error_message