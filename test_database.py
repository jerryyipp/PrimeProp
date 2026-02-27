from src.database import DatabaseManager


def main():
    print("Initializing Database...")
    db = DatabaseManager()

    print("Logging simulated pre-game picks...")
    # Simulate the script finding good bets
    db.log_pick("LeBron James", "Points", 25.5, 28.0, 0.12, "Over")
    db.log_pick("Stephen Curry", "Threes", 4.5, 3.0, -0.25, "Under")
    db.log_pick("Nikola Jokic", "Rebounds", 12.5, 14.1, 0.08, "Over")

    print("Simulating next-day results update...")
    # Update the SQLite table manually to simulate grading the bets
    with db._conn:
        db._conn.execute("UPDATE picks SET actual_result = 30.0, won = 1 WHERE player_name = 'LeBron James'")
        db._conn.execute("UPDATE picks SET actual_result = 5.0, won = 0 WHERE player_name = 'Stephen Curry'")
        db._conn.execute("UPDATE picks SET actual_result = 15.0, won = 1 WHERE player_name = 'Nikola Jokic'")

    print("\n--- Performance Tracking ---")
    total, wins, losses, win_pct = db.get_win_rate()
    print(f"Total Graded Picks: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_pct}%")


if __name__ == "__main__":
    main()
