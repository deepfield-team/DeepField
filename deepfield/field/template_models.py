"""String templates for t-Navigator DATA file."""

ORTHOGONAL_GRID = \
'''MAPAXES
$mapaxes
/

DX
$dx /

DY
$dy /

DZ
$dz /

TOPS
$tops /'''

CORNERPOINT_GRID = \
'''INCLUDE
'$zcorn' /
/

INCLUDE
'$coord' /
/'''

DEFAULT_TN_MODEL = \
'''RUNSPEC

TITLE
$title

MULTOUT
MULTOUTS

START
$start
/

$units

$phases

DIMENS
$dimens /

RUNCTRL
WELLEQUATIONS 1 /
WATERZONE 1 /
/
TNAVCTRL
LONGNAMES 1 /
/

GRID

$grid_specs

INCLUDE
'$actnum' /

$aquifers

$rock_grid

EDIT

PROPS

--ROCK

$tables

$rock_props

SOLUTION

$states

SUMMARY

SEPARATE
TCPU

WBHP
	"*"
/
WOPR
	"*"
/
WWPR
	"*"
/
WGPR
	"*"
/
WWIR
	"*"
/

RECU

$dates

RATE 1 MONTH WELL EXAC END FIEL GROU CRAT LRAT /

FREQ 0 0 1
/

-------------------------------welltracks------------------------------------
TFIL
'$welltrack' /

--------------------------------events---------------------------------------
INCLUDE
'$perf' /

INCLUDE
'$events' /

INCLUDE
'$group' /

END
'''

DEFAULT_ECL_MODEL = \
'''RUNSPEC

TITLE
$title

MULTOUT
MULTOUTS

START
$start
/

$units

$phases

DIMENS
$dimens /

RUNCTRL
WELLEQUATIONS 1 /
WATERZONE 1 /
/
TNAVCTRL
LONGNAMES 1 /
/

GRID

$grid_specs

INCLUDE
'$actnum' /

INCLUDE
'$faults' /

INCLUDE
'$multflt' /

$aquifers

$rock_grid

EDIT

PROPS

--ROCK

$tables

$rock_props

SOLUTION

$states

SUMMARY

SEPARATE
TCPU

WBHP
	"*"
/
WOPR
	"*"
/
WWPR
	"*"
/
WGPR
	"*"
/
WWIR
	"*"
/

SCHEDULE

INCLUDE
'$gruptree' /

INCLUDE
'$welspecs' /

INCLUDE
'$schedule' /

END
'''
