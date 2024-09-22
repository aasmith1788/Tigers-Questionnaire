#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Ellipse

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def create_pitcher_dashboard(pitcher_data, pitcher_name):
    # Set style and figure parameters
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.color'] = 'gray'
    
    fig = plt.figure(figsize=(24, 24))  # Increased figure height
    gs = GridSpec(4, 3, figure=fig, height_ratios=[0.2, 1, 1.2, 0.4])  # Added a row for formulas
    
    # Add a black border to the figure
    fig.patch.set_linewidth(2)  # Set the width of the border
    fig.patch.set_edgecolor('black')  # Set the color of the border
    
    # Color map for pitch types
    pitch_type_colors = {
        'FF': 'blue',          # 4-Seam Fastball
        'SI': 'lightblue',      # Sinker
        'FC': 'purple',         # Cutter
        'CH': 'orange',         # Changeup
        'FS': 'gold',           # Split-finger
        'FO': 'darkgoldenrod',  # Forkball
        'SC': 'lightgreen',     # Screwball
        'CU': 'green',          # Curveball
        'KC': 'darkgreen',      # Knuckle Curve
        'CS': 'forestgreen',    # Slow Curve
        'SL': 'red',            # Slider
        'ST': 'darkred',        # Sweeper
        'SV': 'maroon'          # Slurve
    }

    # Count stats table (top row)
    ax_count_stats = fig.add_subplot(gs[0, :])
    count_stats_table(pitcher_data, ax_count_stats, fontsize=25)
    
    # Pitch location plot for LHH
    ax1 = fig.add_subplot(gs[1, 0])
    create_pitch_location_plot(pitcher_data[pitcher_data['BatterSide'] == 'L'], ax1, 'LHH', pitch_type_colors)
    
    # Pitch location plot for RHH
    ax2 = fig.add_subplot(gs[1, 1])
    create_pitch_location_plot(pitcher_data[pitcher_data['BatterSide'] == 'R'], ax2, 'RHH', pitch_type_colors)
    
    # Pitch break plot
    ax3 = fig.add_subplot(gs[1, 2])
    create_pitch_break_plot(pitcher_data, ax3, pitch_type_colors)  # Pass pitch_type_colors here
    
    # Pitch stats table (larger section)
    ax4 = fig.add_subplot(gs[2, :])
    pitch_table(pitcher_data, ax4, fontsize=25)
    
    # Formula display (bottom section)
    ax5 = fig.add_subplot(gs[3, :])
    display_formulas(ax5)
    
    # Determine whether the pitcher is LHP or RHP
    pitcher_hand = pitcher_data['PitcherHand'].iloc[0]  # Assuming all rows have the same handedness
    hand_label = "LHP" if pitcher_hand == "L" else "RHP"
    
    # Main title with handedness included
    fig.suptitle(f"{pitcher_name} ({hand_label}) - Pitch Analysis", fontsize=35, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4)
    plt.show()


    
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

def create_pitch_location_plot(data, ax, title, pitch_type_colors):
    # Get average pitch locations
    avg_locations = data.groupby('PitchType').agg(
        avg_x=('TrajectoryLocationX', 'mean'),
        avg_z=('TrajectoryLocationZ', 'mean')
    ).reset_index()

    # Plot each pitch type with predefined colors
    for _, row in avg_locations.iterrows():
        ax.scatter(row['avg_x'], row['avg_z'], 
                   label=row['PitchType'], 
                   alpha=0.7, 
                   s=400, 
                   color=pitch_type_colors.get(row['PitchType'], 'gray'))

    # Add strike zone
    zone_bottom = data['StrikeZoneBottom'].mean()
    zone_top = data['StrikeZoneTop'].mean()
    zone_width = 1.4166
    zone_height = zone_top - zone_bottom
    zone_center = 0
    zone = Rectangle((zone_center - zone_width / 2, zone_bottom), zone_width, zone_height, 
                     fill=False, color='k', linewidth=2)
    ax.add_patch(zone)

    # Add home plate
    plate_width = zone_width
    plate = plt.Polygon([(-plate_width / 2, 0), (0, 0), (plate_width / 2, 0)], color='k')
    ax.add_patch(plate)

    # Set plot limits with padding
    padding_x = 0.5
    padding_y = 0.5
    ax.set_xlim((zone_center - zone_width / 2) - padding_x, (zone_center + zone_width / 2) + padding_x)
    ax.set_ylim(zone_bottom - padding_y, zone_top + padding_y)

    # Set axis labels, title, and legend
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('', fontsize=18)
    ax.set_title(f'Average Pitch Locations vs {title}', fontsize=20)
    
    # Improve legend readability
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Pitch Type", loc="upper right", fontsize=14, title_fontsize='16')
    ax.invert_xaxis()

    # Remove ticks and background grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')
    ax.grid(False)

    # Hide plot spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set the aspect ratio
    ax.set_aspect(1.02)

# Call the function with the new color map
# create_pitch_location_plot(data, ax, 'Left-handed Batters', pitch_type_colors)
def create_pitch_break_plot(data, ax, pitch_type_colors):
    # Convert data from feet to inches
    data['TrajectoryHorizontalBreak'] *= 12
    data['TrajectoryVerticalBreakInduced'] *= 12
    
    # Loop through each pitch type and plot using color from the pitch_type_colors dictionary
    for pitch_type in data['PitchType'].unique():
        pitch_data = data[data['PitchType'] == pitch_type]
        ax.scatter(pitch_data['TrajectoryHorizontalBreak'], 
                   pitch_data['TrajectoryVerticalBreakInduced'], 
                   label=pitch_type, 
                   alpha=0.7, 
                   s=70, 
                   color=pitch_type_colors.get(pitch_type, 'gray'))  # Use 'gray' as fallback if pitch type is missing in the dictionary
    
    # Set labels and title
    ax.set_xlabel('Horizontal Break (in)', fontsize=26)
    ax.set_ylabel('Vertical Break (in)', fontsize=26)
    ax.set_title('Pitch Break', fontsize=28)
    
    # Set legend and improve readability
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Pitch Type", loc="upper right", fontsize=14, title_fontsize='16')
    
    # Add dashed lines for reference
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    # Set x and y limits with padding
    max_horz = data['TrajectoryHorizontalBreak'].max() * 1.1
    min_horz = data['TrajectoryHorizontalBreak'].min() * 1.1
    max_vert = data['TrajectoryVerticalBreakInduced'].max() * 1.1
    min_vert = data['TrajectoryVerticalBreakInduced'].min() * 1.1
    
    ax.set_xlim(min_horz, max_horz)
    ax.set_ylim(min_vert, max_vert)
    
    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

def count_stats_table(pitcher_data, ax, fontsize=25):
    batters_faced = pitcher_data['AtBatNumber'].nunique()
    strikeouts = (pitcher_data['PitchCall'] == 'strikeout').sum()
    walks = (pitcher_data['PitchCall'] == 'walk').sum()
    singles = (pitcher_data['PitchCall'] == 'single').sum()
    doubles = (pitcher_data['PitchCall'] == 'double').sum()
    triples = (pitcher_data['PitchCall'] == 'triple').sum()
    home_runs = (pitcher_data['PitchCall'] == 'home_run').sum()

    hits = singles + doubles + triples + home_runs
    at_bats = batters_faced - walks - (pitcher_data['PitchCall'] == 'hit_by_pitch').sum()

    total_bases = singles + (2 * doubles) + (3 * triples) + (4 * home_runs)

    valid_pitches = pitcher_data['PitchType'].notna().sum()

    count_stats = {
        'Pitches': valid_pitches,
        'BF': batters_faced,
        'Ks': strikeouts,
        'BBs': walks,
        'K%': f"{(strikeouts / batters_faced * 100):.1f}%" if batters_faced > 0 else "0.0%",
        'BB%': f"{(walks / batters_faced * 100):.1f}%" if batters_faced > 0 else "0.0%",
        '1Bs': singles,
        '2Bs': doubles,
        '3Bs': triples,
        'HRs': home_runs,
        'Opp AVG': f"{(hits / at_bats):.3f}" if at_bats > 0 else "0.000",
        'Opp SLG': f"{(total_bases / at_bats):.3f}" if at_bats > 0 else "0.000"
    }
    
    outs = (
        strikeouts +
        (pitcher_data['PitchCall'] == 'field_out').sum() +
        (pitcher_data['PitchCall'] == 'sac_bunt').sum() +
        (pitcher_data['PitchCall'] == 'force_out').sum() +
        2 * (pitcher_data['PitchCall'] == 'grounded_into_double_play').sum()
    )
    
    innings_pitched = f"{outs // 3}"
    if outs % 3 == 1:
        innings_pitched += " 1/3"
    elif outs % 3 == 2:
        innings_pitched += " 2/3"
    count_stats['IP'] = innings_pitched

    df_counts = pd.DataFrame([count_stats])
    table = ax.table(cellText=df_counts.values, colLabels=df_counts.columns, 
                     cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.2, 1.2)
    for i in range(len(df_counts) + 1):
        for j in range(len(count_stats)):
            cell = table.get_celld()[(i, j)]
            if i == 0:
                cell.set_facecolor('#333333')
                cell.set_text_props(color='white', fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('#FFFFFF')
    ax.axis('off')

def pitch_table(pitcher_data, ax, fontsize= 25):
    pitcher_data['true_spin'] = np.sqrt(
        pitcher_data['SpinVectorX']**2 + 
        pitcher_data['SpinVectorZ']**2
    )
    pitcher_data['spin_efficiency'] = (pitcher_data['true_spin'] / pitcher_data['ReleaseSpinRate']) * 100
    pitcher_data['spin_efficiency'] = np.clip(pitcher_data['spin_efficiency'], 0, 100)
    
    # Define outcomes that count as at-bats and hits
    at_bat_outcomes = ['single', 'double', 'triple', 'home_run', 'field_out', 'strikeout', 
                       'swinging_strike', 'called_strike', 'foul', 'foul_bunt']
    hit_outcomes = ['single', 'double', 'triple', 'home_run']
    
    # Group by PitchType and calculate stats
    pitch_stats = pitcher_data.groupby('PitchType').agg(
        valid_pitches=('PitchType', 'count'),
        at_bats=('PitchCall', lambda x: sum(x.isin(at_bat_outcomes))),
        hits=('PitchCall', lambda x: sum(x.isin(hit_outcomes))),
        singles=('PitchCall', lambda x: sum(x == 'single')),
        doubles=('PitchCall', lambda x: sum(x == 'double')),
        triples=('PitchCall', lambda x: sum(x == 'triple')),
        home_runs=('PitchCall', lambda x: sum(x == 'home_run')),
        release_speed=('ReleaseSpeed', 'mean'),
        vert_break=('TrajectoryVerticalBreakInduced', 'mean'),
        horz_break=('TrajectoryHorizontalBreak', 'mean'),
        release_spin_rate=('ReleaseSpinRate', 'mean'),
        spin_efficiency=('spin_efficiency', 'mean'),
        release_pos_x=('ReleasePositionX', 'mean'),
        release_pos_z=('ReleasePositionZ', 'mean'),
        release_extension=('ReleaseExtension', 'mean'),
        whiff_percentage=('PitchCall', lambda x: sum(x == 'swinging_strike') / len(x)),
    ).reset_index()
    
    # Calculate batting average and slugging percentage
    pitch_stats['total_bases'] = (pitch_stats['singles'] + 
                                  2 * pitch_stats['doubles'] + 
                                  3 * pitch_stats['triples'] + 
                                  4 * pitch_stats['home_runs'])
    pitch_stats['opp_ba'] = pitch_stats['hits'] / pitch_stats['at_bats']
    pitch_stats['opp_slg'] = pitch_stats['total_bases'] / pitch_stats['at_bats']
    
    total_pitches = pitch_stats['valid_pitches'].sum()
    pitch_stats['pitch_usage'] = pitch_stats['valid_pitches'] / total_pitches
    
    df_plot = pitch_stats.copy()
    df_plot['Count'] = df_plot['valid_pitches'].astype(int)
    df_plot['Usage'] = df_plot['pitch_usage'].apply(lambda x: f"{x:.1%}")
    df_plot['Velo'] = df_plot['release_speed'].apply(lambda x: f"{x:.1f}")
    df_plot['V Break'] = df_plot['vert_break'].apply(lambda x: f"{x:.1f}")
    df_plot['H Break'] = df_plot['horz_break'].apply(lambda x: f"{x:.1f}")
    df_plot['Spin'] = df_plot['release_spin_rate'].apply(lambda x: f"{x:.0f}")
    df_plot['SpinEff%'] = df_plot['spin_efficiency'].apply(lambda x: f"{x:.1f}%")
    df_plot['hRel'] = df_plot['release_pos_x'].apply(lambda x: f"{x:.2f}")
    df_plot['vRel'] = df_plot['release_pos_z'].apply(lambda x: f"{x:.2f}")
    df_plot['Ext'] = df_plot['release_extension'].apply(lambda x: f"{x:.2f}")
    df_plot['OppBA'] = df_plot.apply(lambda x: f"{x['opp_ba']:.3f}" if x['at_bats'] > 0 else "N/A", axis=1)
    df_plot['OppSLG'] = df_plot.apply(lambda x: f"{x['opp_slg']:.3f}" if x['at_bats'] > 0 else "N/A", axis=1)
    df_plot['Whiff%'] = df_plot['whiff_percentage'].apply(lambda x: f"{x:.1%}")
    
    df_final = df_plot[['PitchType', 'Count', 'Usage', 'Velo', 'V Break', 'H Break', 'Spin', 
                        'SpinEff%', 'hRel', 'vRel', 'Ext', 'OppBA', 'OppSLG', 'Whiff%']]
    
    table = ax.table(cellText=df_final.values, colLabels=df_final.columns, 
                     cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.8)
    
    for i in range(len(df_final) + 1):
        for j in range(len(df_final.columns)):
            cell = table.get_celld()[(i, j)]
            if i == 0:
                cell.set_facecolor('#333333')
                cell.set_text_props(color='white', fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('#FFFFFF')
    
    ax.axis('off')

import textwrap

def display_formulas(ax):
    formulas = [
        r"Opp BA = $\frac{Hits}{At Bats}$",
        r"Opp SLG = $\frac{Total Bases}{At Bats}$",
        r"Whiff% = $\frac{Swinging Strikes}{Total Pitches}$",
        r"Spin Efficiency = $\frac{\sqrt{SpinVectorX^2 + SpinVectorZ^2}}{ReleaseSpinRate} \times 100\%$"
    ]
    
    explanations = [
        "Hits: Number of successful hits (singles, doubles, triples, home runs)",
        "At Bats: Number of pitches labeled as single, double, triple, home_run, field_out, strikeout, swinging_strike, called_strike, foul, foul_bunt",
        "Total Bases: Sum of bases from hits (1 for single, 2 for double, 3 for triple, 4 for home run)",
        "Swinging Strikes: Number of pitches labeled as swinging_strike (strikeout not included)",
        "Total Pitches: Total number of pitches thrown",
        "SpinVectorX, SpinVectorZ: Components of the spin vector",
        "ReleaseSpinRate: Total spin rate at ball release"
    ]
    
    ax.axis('off')  # Turn off the axis
    
    formula_text = "Formulas: " + " | ".join(formulas)
    explanation_text = "Variable Explanations: " + " | ".join(explanations)
    
    full_text = formula_text + "\n" + explanation_text
    
    ax.text(0, 0, full_text, ha='left', va='bottom', fontsize=25, 
            transform=ax.transAxes, wrap=True)
# Load the data
data = pd.read_csv(r"C:\Users\aasmi\Downloads\AnalyticsQuestionnairePitchData.csv")

# Create dashboards for all unique pitchers
for pitcher_id in data['PitcherId'].unique():
    pitcher_data = data[data['PitcherId'] == pitcher_id]
    if not pitcher_data.empty:
        create_pitcher_dashboard(pitcher_data, f"Pitcher {pitcher_id}")
    else:
        print(f"No data available for Pitcher {pitcher_id}")


# In[ ]:





# In[ ]:




