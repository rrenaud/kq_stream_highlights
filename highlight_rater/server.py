"""
Flask server for the highlight rating tool.
"""

import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from pathlib import Path

from . import db
from .models import ChaptersData, list_chapters_files, RATING_LABELS, POSITION_NAMES, format_time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'highlight-rater-dev-key'

# Cache for loaded chapters data
_chapters_cache: dict[str, ChaptersData] = {}


def get_chapters_data(chapters_file: str) -> ChaptersData:
    """Get chapters data, using cache."""
    if chapters_file not in _chapters_cache:
        _chapters_cache[chapters_file] = ChaptersData.load(chapters_file)
    return _chapters_cache[chapters_file]


# ============== Web Routes ==============

@app.route('/')
def index():
    """Home page - select rater and chapters file."""
    raters = db.list_raters()
    chapters_files = list_chapters_files()
    sessions = db.list_sessions()
    return render_template('index.html',
                           raters=raters,
                           chapters_files=chapters_files,
                           sessions=sessions)


@app.route('/rate')
def rate():
    """Rating interface."""
    rater_name = request.args.get('rater')
    chapters_file = request.args.get('chapters')

    if not rater_name or not chapters_file:
        return redirect(url_for('index'))

    # Get or create rater
    rater_id = db.get_or_create_rater(rater_name)

    # Get or create session
    session = db.get_incomplete_session(rater_id, chapters_file)
    if not session:
        chapters_data = get_chapters_data(chapters_file)
        session_id = db.create_session(rater_id, chapters_file, chapters_data.video_id)
        # Load events into algorithm_scores table
        db.load_events_from_chapters(chapters_file, {
            'chapters': [
                {
                    'game_id': g.game_id,
                    'player_events': [e.to_dict() for e in chapters_data.get_events_for_game(g.game_id)]
                }
                for g in chapters_data.games.values()
            ]
        })
    else:
        session_id = session['id']

    return render_template('rater.html',
                           rater_name=rater_name,
                           chapters_file=chapters_file,
                           session_id=session_id,
                           rating_labels=RATING_LABELS,
                           position_names=POSITION_NAMES)


# ============== API Routes ==============

@app.route('/api/raters', methods=['GET', 'POST'])
def api_raters():
    """List or create raters."""
    if request.method == 'POST':
        data = request.json
        rater_id = db.get_or_create_rater(data['name'])
        return jsonify({'id': rater_id, 'name': data['name']})
    return jsonify(db.list_raters())


@app.route('/api/chapters')
def api_chapters():
    """List available chapters files."""
    return jsonify(list_chapters_files())


@app.route('/api/sessions', methods=['GET', 'POST'])
def api_sessions():
    """List or create sessions."""
    if request.method == 'POST':
        data = request.json
        rater_id = db.get_or_create_rater(data['rater_name'])
        chapters_data = get_chapters_data(data['chapters_file'])
        session_id = db.create_session(rater_id, data['chapters_file'], chapters_data.video_id)
        return jsonify({'session_id': session_id})

    rater_id = request.args.get('rater_id', type=int)
    return jsonify(db.list_sessions(rater_id))


@app.route('/api/sessions/<int:session_id>')
def api_session(session_id: int):
    """Get session details."""
    session = db.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    chapters_data = get_chapters_data(session['chapters_file'])
    rated_ids = db.get_rated_event_ids(session_id)

    return jsonify({
        'session': session,
        'total_events': len(chapters_data.events),
        'rated_count': len(rated_ids),
        'video_id': chapters_data.video_id,
    })


@app.route('/api/sessions/<int:session_id>/next')
def api_next_event(session_id: int):
    """Get the next unrated event for a session."""
    session = db.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    chapters_data = get_chapters_data(session['chapters_file'])
    rated_ids = db.get_rated_event_ids(session_id)
    unrated = chapters_data.get_unrated_events(rated_ids)

    if not unrated:
        # All events rated - mark session complete
        db.complete_session(session_id)
        return jsonify({
            'complete': True,
            'total': len(chapters_data.events),
            'rated': len(rated_ids),
        })

    event = unrated[0]
    game = chapters_data.games.get(event.game_id)

    # Get user names for positions
    position_users = {}
    if game:
        for pos, user_id in game.users.items():
            user_info = chapters_data.users.get(int(user_id), {})
            position_users[pos] = user_info.get('name', f'User {user_id}')

    return jsonify({
        'complete': False,
        'event': event.to_dict(),
        'game': game.to_dict() if game else None,
        'position_users': position_users,
        'video_id': chapters_data.video_id,
        'progress': {
            'rated': len(rated_ids),
            'total': len(chapters_data.events),
            'remaining': len(unrated),
        },
        'formatted_time': format_time(event.time),
    })


@app.route('/api/sessions/<int:session_id>/ratings', methods=['GET', 'POST'])
def api_ratings(session_id: int):
    """Get or create ratings for a session."""
    if request.method == 'POST':
        data = request.json
        rating_id = db.create_rating(
            session_id=session_id,
            event_id=data['event_id'],
            game_id=data['game_id'],
            rating=data['rating'],
            response_time_ms=data.get('response_time_ms'),
        )
        return jsonify({'rating_id': rating_id})

    return jsonify(db.get_ratings_for_session(session_id))


@app.route('/api/sessions/<int:session_id>/complete', methods=['POST'])
def api_complete_session(session_id: int):
    """Mark a session as complete."""
    db.complete_session(session_id)
    return jsonify({'success': True})


def run_server(host='127.0.0.1', port=5001, debug=True):
    """Run the Flask development server."""
    print(f"Starting highlight rater server at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
