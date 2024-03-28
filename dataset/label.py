# First, we need to get the list of games somehow. We found out that Twitch statistics 
# might be usable, but we should not rely only on it. To get the list of games, here's how
# we will aproach it. We also think that 30 games is a good number to start with.

# 0 - 9: Top games by current viewership on Twitch 
# 10 - 17: Top games by all time peak viewership on Twitch, that not already in the list
# 18 - 25: Top games by all time peak players count on Steam, that not already in the list
# 26 - 29: The game that we want think is important, sourcing from Google Search "Best FPS games"
# To verify game is FPS or not, we check it with Wikipedia's list of FPS games, we will also use it for naming the games:
# https://en.wikipedia.org/wiki/List_of_first-person_shooters

games_list = [
    # Top games by current viewership on Twitch, https://twitchtracker.com/games
    "VALORANT", #0
    "Counter-Strike: Global Offensive", #1, Twitch combines 2 games into 1 despite being different enough
    "Counter-Strike 2", #2 
    "Overwatch 2", #3
    "Call of Duty: Warzone", #4
    "Tom Clancy's Rainbow Six: Siege", #5
    "Apex Legends", #6
    "Escape from Tarkov", #7
    "Rust", #8
    "PUBG: BATTLEGROUNDS", #9
    
    # Top games by all time peak viewership on Twitch, https://twitchtracker.com/games/peak-viewers
    # Too many Call of Duty games. Might be harder to split. But they all passed the treashold, so we will do nothing
    "Overwatch", #10
    "Cyberpunk 2077", #11
    "Call of Duty: Modern Warfare II", #12
    "Call of Duty: Black Ops Cold War", #13
    "Call of Duty: Black Ops 4", #14
    "Dying Light 2: Stay Human", #15
    "Destiny 2", #16
    "Call of Duty: Modern Warfare III", #17
    
    # Top games by all time peak players count on Steam, https://steamdb.info/charts/?sort=peak
    "Fallout 4", #18
    "Halo Infinite", #19
    "Team Fortress 2", #20
    "PAYDAY 2", #21
    "Left 4 Dead 2", #22
    "Battlefield 2042", #23
    "Borderlands 2", #24
    "Battlefield V", #25
    
    # The game that we want think is important, sourcing from Google Search "Best FPS games"
    # The game should consistenly appear in the lists
    # PG: https://www.pcgamer.com/best-fps-games/
    # USA: https://ftw.usatoday.com/lists/best-fps-games
    # GRD: https://www.gamesradar.com/best-fps-games/
    # RPS: https://www.rockpapershotgun.com/best-fps-games
    # GR: https://gamerant.com/best-first-person-shooters/
    "Titanfall 2" #26   | PG: Yes, USA: #9,                   GRD: #7,  RPS: #22, GR: #11
    "DOOM Eternal", #27 | PG: Yes, USA: No (but Doom #4),     GRD: #3,  RPS: #11, GR: No, but doom at #9
    "Half-Life 2", #28  | PG: No,  USA: No (but Half-Life #5),GRD: #20, RPS: #5,  GR: #3
    "Metro Exodus" #29  | PG: Yes, USA: #15,                  GRD: #11, RPS: No,  GR: No
]