# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: setting.py
@Time: 2018/5/17 下午9:02
@Description:
"""
FEATURE1 = [
    'Ratio_Size', 'Ratio_ROA', 'Ratio_Lev', 'Ratio_Inventory', 'Ratio_Mgt_change',
    'Ratio_Relcomp', 'Ratio_PID', 'Ratio_MgmtO', 'Ratio_AuditFee', 'Ratio_INST_CON',
    'Ratio_First', 'Ratio_Gap', 'Ratio_Distress', 'Raw_MA', 'Raw_Dual',
    'Raw_H10', 'Raw_SOE', 'Raw_SM', 'Raw_Crosslist', 'Raw_Auditor',
    'Raw_Auditor_resigned', 'Raw_Restate', 'Raw_IC_audit', 'Raw_Segments', 'Raw_Foreign_transactions',
    'Raw_Loss', 'Raw_Age', 'year', 'Industry'
]
FEATURE1_NEW = [
    'Raw_Segments', 'Raw_MA', 'Ratio_Size', 'Raw_Loss', 'Ratio_Distress',
    'Ratio_ROA', 'Ratio_Lev', 'Raw_Age', 'Ratio_Mgt_change', 'Raw_Dual',
    'Ratio_PID', 'Ratio_MgmtO', 'Raw_Auditor_resigned', 'Raw_Auditor', 'Ratio_AuditFee',
    'Raw_IC_audit', 'Ratio_INST_CON', 'Ratio_First', 'Raw_H10', 'Raw_Restate',
    'Raw_Crosslist', 'Raw_SOE', 'year', 'Industry'
]
FEATURE2 = ['Raw_MA', 'Raw_Dual', 'Raw_H10', 'Raw_SOE', 'Raw_SM',
            'Raw_Crosslist', 'Raw_Auditor', 'Raw_Auditor_resigned', 'Raw_Restate', 'Raw_IC_audit',
            'Raw_Segments', 'Raw_Foreign_transactions', 'Raw_Loss', 'Raw_Age', 'Raw_invt',
            'Raw_ca', 'Raw_at', 'Raw_cl', 'Raw_lt', 'Raw_re',
            'Raw_sale', 'Raw_fe', 'Raw_ebt', 'Raw_ni', 'Raw_prcc_f',
            'Raw_Board', 'Raw_id', 'Raw_mgmtsh', 'Raw_first', 'Raw_otherblock',
            'Raw_csho', 'Raw_instsh', 'Raw_chairchg', 'Raw_CEOchg', 'Raw_CEOcomp',
            'Raw_othercomp', 'Raw_audit_fee', 'year', 'Industry'
            ]
FEATURE2_NEW = ['Raw_MA', 'Raw_Dual', 'Raw_SOE', 'Raw_Crosslist', 'Raw_Auditor',
                'Raw_Auditor_resigned', 'Raw_Restate', 'Raw_IC_audit', 'Raw_Loss', 'Raw_H10',
                'Raw_Segments', 'Raw_Age', 'Raw_chairchg', 'Raw_CEOchg', 'Raw_ca',
                'Raw_at', 'Raw_cl', 'Raw_lt', 'Raw_re', 'Raw_sale',
                'Raw_fe', 'Raw_ebt', 'Raw_ni', 'Raw_prcc_f', 'Raw_Board',
                'Raw_id', 'Raw_mgmtsh', 'Raw_first', 'Raw_csho', 'Raw_instsh',
                'Raw_audit_fee', 'year', 'Industry'
                ]
USER = 'root'
PASSWORD = 'SELECT * FROM users;'
HOST = "sis2"
# YEAR_TO_INDEX = {'2006': 0, '2007': 1, '2008': 2, '2009': 3, '2010': 4,
#                  '2011': 5, '2012': 6, '2013': 7, '2014': 8, '2015': 9}
INDUSTRY_TO_INDEX = {'K': 0, 'C27': 1, 'N': 2, 'H': 3, 'C37': 4,
                     'S': 5, 'E': 6, 'C30': 7, 'C39': 8, 'C15': 9,
                     'G': 10, 'F': 11, 'D': 12, 'C36': 13, 'L': 14,
                     'C33': 15, 'C13': 16, 'C38': 17, 'C25': 18, 'C32': 19,
                     'Q': 20, 'C26': 21, 'R': 22, 'C35': 23, 'C17': 24,
                     'C34': 25, 'C28': 26, 'B': 27, 'C22': 28, 'I': 29,
                     'C29': 30, 'P': 31, 'C31': 32, 'C41': 33, 'A': 34,
                     'C21': 35, 'C14': 36, 'C23': 37, 'C42': 38, 'C20': 39,
                     'C18': 40, 'C40': 41, 'C24': 42, 'M': 43, 'C19': 44}
