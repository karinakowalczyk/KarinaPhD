luparams = {'snes_monitor':None,
    'mat_type':'aij',
    'ksp_type':'preonly',
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps'}

sparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "snes_converged_reason": None,
    "ksp_type": "fgmres",
    "ksp_gmres_modifiedgramschmidt": None,
    "ksp_converged_reason": None,
    'ksp_monitor': None,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_0_fields": "0,2,3",
    "pc_fieldsplit_1_fields": "1",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

sparameters_exact = { "mat_type": "aij",
                   'snes_monitor': None,
                   #'snes_view': None,
                   #'snes_type' : 'ksponly',
                   'ksp_monitor_true_residual': None,
                   'snes_converged_reason': None,
                   'ksp_converged_reason': None,
                   "ksp_type" : "preonly",
                   "pc_type" : "lu",
                   "pc_factor_mat_solver_type": "mumps"
                   }


topleft_LU = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

topleft_LS = {
    'ksp_type': 'preonly',
    'pc_type': 'python',
    "pc_python_type": "firedrake.AssembledPC",
    'assembled_pc_type': 'python',
    'assembled_pc_python_type': 'firedrake.ASMStarPC',
    "assembled_pc_star_sub_pc_type": "lu",
    'assembled_pc_star_dims': '0',
    'assembled_pc_star_sub_pc_factor_mat_solver_type' : 'mumps'
    #'assembled_pc_linesmooth_star': '1'
}
bottomright = {
    "ksp_type": "gmres",
    "ksp_max_it": 3,
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
    "Mp_pc_type": "bjacobi",
    "Mp_sub_pc_type": "ilu"
}

sparameters["fieldsplit_1"] = bottomright


sparameters["fieldsplit_0"] = topleft_LS


sparameters_mg = {
        "snes_monitor": None,
        "mat_type": "aij",
        "ksp_type": "fgmres",
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "ksp_atol": 1e-8,
        "ksp_rtol": 1e-8,
        "ksp_max_it": 400,
        "pc_type": "mg",
        "pc_mg_cycle_type": "v",
        "pc_mg_type": "multiplicative",
        "mg_levels_ksp_type": "gmres",
        "mg_levels_ksp_max_it": 3,
        "mg_levels_pc_type": "python",
        'mg_levels_pc_python_type': 'firedrake.ASMStarPC',
        "mg_levels_pc_star_sub_pc_type": "lu",
        'mg_levels_pc_star_dims': '0',
        'mg_levels_pc_star_sub_pc_factor_mat_solver_type' : 'mumps',
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
        "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    }






    piparams_exact = {"ksp_type": "preonly",
                  "ksp_monitor": None,
                  #"ksp_view":None,
                  'pc_type':'lu',
                  'pc_factor_mat_solver_type':'mumps'
                  }