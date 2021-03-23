from flask import Flask
from flask_restplus import Api, Resource
from werkzeug.datastructures import FileStorage
from PIL import Image
import base64
from io import BytesIO
from dehaze import Dehazer
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

dehazer = Dehazer()


upload_parser = api.parser()
upload_parser.add_argument('image', location='files',
                           type=FileStorage, required=True,
                           help="Base64 encoded Image String")


@api.route('/dehaze')
class Dehaze(Resource):

    @api.expect(upload_parser)
    def post(self):
        args = upload_parser.parse_args()
        input_byte_array = args['image'].read()

        try:
            img = Image.open(BytesIO(input_byte_array))
            dehazed_img = dehazer.dehaze(img)

            im_file = BytesIO()
            dehazed_img.save(im_file, format="JPEG")

            output_byte_string = im_file.getvalue()
            output_byte_string = f'data:image/png;base64,{base64.b64encode(output_byte_string).decode()}'
            return {'success': True, 'out_img': output_byte_string}, 200

        except Exception as e:
            print(e)
            return {'success': False}, 422


if __name__ == "__main__":
    app.run(debug=True)
